"""The only entry point from the Internet, in front of vLLM.

vLLM's `--api-key` only protects /v1, /v2, /inference and /cohere; other paths such as
/invocations (same text generation as /v1) or /sleep stay open. This proxy:
- checks the key on every path: `Authorization: Bearer <key>` (OpenAI) or `x-api-key: <key>` (Anthropic);
- lets through only the model APIs, everything else gets 404;
- logs nothing and forwards responses chunk by chunk (streaming still works).

    python proxy.py <listen port> <vLLM port> <api key>
"""

import secrets
import sys

from aiohttp import ClientSession, ClientTimeout, web

ALLOWED = {
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/responses"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("POST", "/tokenize"),
    ("POST", "/detokenize"),
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


def make_app(upstream: str, key: str) -> web.Application:
    async def handle(request: web.Request) -> web.StreamResponse:
        if (request.method, request.path) not in ALLOWED:
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
    web.run_app(make_app(f"http://127.0.0.1:{vllm_port}", api_key),
                host="127.0.0.1", port=listen, print=None, access_log=None)
