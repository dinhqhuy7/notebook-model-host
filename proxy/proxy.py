"""The only entry point from the Internet, in front of vLLM.

vLLM's `--api-key` only protects /v1, /v2, /inference and /cohere; other paths such as
/invocations (same text generation as /v1) or /sleep stay open. This proxy:
- checks the key on every path: `Authorization: Bearer <key>` (OpenAI) or `x-api-key: <key>` (Anthropic);
- lets through only the model APIs of the chosen profile, everything else gets 404;
- logs nothing and forwards responses chunk by chunk (streaming still works).

    python proxy.py <listen port> <vLLM port> <api key> [profile]

The profile says which APIs the model has: "chat" (default: language, vision and multimodal models), "retrieval"
(scoring, reranking and embedding models) or "audio" (speech recognition and generation). A model gets all the APIs
of its profile; a new kind of model that needs other paths gets a new profile here.
"""

import secrets
import sys

from aiohttp import ClientSession, ClientTimeout, web

COMMON = {("GET", "/v1/models"), ("POST", "/tokenize"), ("POST", "/detokenize")}
PROFILES = {
    "chat": COMMON | {("POST", path) for path in (
        "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens")},
    "retrieval": COMMON | {("POST", path) for path in (
        "/score", "/v1/score", "/rerank", "/v1/rerank", "/v2/rerank", "/v1/embeddings", "/v2/embed", "/pooling")},
    "audio": COMMON | {("POST", path) for path in (
        "/v1/audio/transcriptions", "/v1/audio/translations", "/v1/audio/speech")},
}
# Headers to forward; all others (Host, Cookie...) are dropped.
FORWARD_UP = ("Content-Type", "Accept", "anthropic-version", "anthropic-beta")
FORWARD_DOWN = ("Content-Type", "Cache-Control")
SESSION = web.AppKey("session", ClientSession)


def given_key(request: web.Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def make_app(upstream: str, key: str, profile: str = "chat") -> web.Application:
    allowed = PROFILES[profile]

    async def handle(request: web.Request) -> web.StreamResponse:
        if (request.method, request.path) not in allowed:
            return web.json_response({"error": "Not found"}, status=404)
        if not secrets.compare_digest(given_key(request).encode(), key.encode()):
            return web.json_response({"error": "Unauthorized"}, status=401)
        headers = {h: request.headers[h] for h in FORWARD_UP if h in request.headers}
        headers["Authorization"] = f"Bearer {key}"
        try:
            upstream_reply = await request.app[SESSION].request(
                request.method, upstream + request.rel_url.path_qs,
                data=await request.read(), headers=headers,
            )
        except Exception as error:  # vLLM is down or restarting
            return web.json_response({"error": f"upstream: {error!r}"}, status=502)
        async with upstream_reply:
            reply = web.StreamResponse(
                status=upstream_reply.status,
                headers={h: upstream_reply.headers[h] for h in FORWARD_DOWN if h in upstream_reply.headers},
            )
            await reply.prepare(request)
            try:
                async for chunk in upstream_reply.content.iter_any():
                    await reply.write(chunk)
                await reply.write_eof()
            except Exception:  # client disconnected midway
                pass
            return reply

    async def start(app: web.Application) -> None:
        app[SESSION] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10, sock_read=900))

    async def stop(app: web.Application) -> None:
        await app[SESSION].close()

    app = web.Application(client_max_size=32 * 2**20)
    app.router.add_route("*", "/{tail:.*}", handle)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    return app


if __name__ == "__main__":
    listen, vllm_port, api_key = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    web.run_app(make_app(f"http://127.0.0.1:{vllm_port}", api_key, sys.argv[4] if len(sys.argv) > 4 else "chat"),
                host="127.0.0.1", port=listen, print=None, access_log=None)
