"""Test the proxy against a fake vLLM, no GPU needed.

    python -m unittest discover -s tests
"""

import json
import sys
import unittest
from pathlib import Path

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy"))
from proxy import make_app  # noqa: E402

KEY = "test-key"


def fake_vllm() -> web.Application:
    """Echoes what the proxy sends, plus one multi-chunk streaming route."""

    async def echo(request: web.Request) -> web.Response:
        return web.json_response({
            "path": request.rel_url.path_qs,
            "auth": request.headers.get("Authorization"),
            "cookie": request.headers.get("Cookie"),
            "body": (await request.read()).decode(),
        })

    async def stream(request: web.Request) -> web.StreamResponse:
        reply = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await reply.prepare(request)
        for i in range(3):
            await reply.write(f"data: {i}\n\n".encode())
        await reply.write(b"data: [DONE]\n\n")
        return reply

    app = web.Application()
    app.router.add_post("/v1/completions", stream)
    app.router.add_route("*", "/{tail:.*}", echo)
    return app


class ProxyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.upstream = TestServer(fake_vllm())
        await self.upstream.start_server()
        self.proxy = TestServer(make_app(str(self.upstream.make_url("")).rstrip("/"), KEY))
        await self.proxy.start_server()
        self.client = ClientSession()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.proxy.close()
        await self.upstream.close()

    async def call(self, method: str, path: str, headers=None, body=None):
        async with self.client.request(
            method, self.proxy.make_url(path), headers=headers or {}, data=body
        ) as reply:
            return reply.status, await reply.read(), reply.headers.get("Content-Type", "")

    async def test_bearer_key_is_forwarded(self) -> None:
        body = json.dumps({"model": "m", "messages": []})
        status, raw, _ = await self.call(
            "POST", "/v1/chat/completions?x=1",
            {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Cookie": "a=b"},
            body,
        )
        self.assertEqual(status, 200)
        seen = json.loads(raw)
        self.assertEqual(seen["path"], "/v1/chat/completions?x=1")
        self.assertEqual(seen["auth"], f"Bearer {KEY}")
        self.assertIsNone(seen["cookie"])
        self.assertEqual(seen["body"], body)

    async def test_anthropic_x_api_key(self) -> None:
        status, raw, _ = await self.call("POST", "/v1/messages", {"x-api-key": KEY}, "{}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["auth"], f"Bearer {KEY}")

    async def test_missing_or_wrong_key(self) -> None:
        for headers in ({}, {"Authorization": "Bearer nope"}, {"x-api-key": "nope"}, {"Authorization": KEY}):
            status, _, _ = await self.call("POST", "/v1/chat/completions", headers, "{}")
            self.assertEqual(status, 401, headers)

    async def test_admin_paths_blocked_even_with_key(self) -> None:
        auth = {"Authorization": f"Bearer {KEY}"}
        for method, path in [
            ("POST", "/invocations"), ("GET", "/server_info"), ("POST", "/reset_prefix_cache"),
            ("POST", "/sleep"), ("POST", "/v1/load_lora_adapter"), ("GET", "/metrics"),
            ("GET", "/v1/chat/completions"), ("POST", "/v1/models"),
            ("GET", "/v1/responses/resp_abc"), ("POST", "/v1/responses/resp_abc/cancel"),
        ]:
            status, _, _ = await self.call(method, path, auth, "{}")
            self.assertEqual(status, 404, path)

    async def test_stream_passes_through(self) -> None:
        status, raw, kind = await self.call("POST", "/v1/completions", {"Authorization": f"Bearer {KEY}"}, "{}")
        self.assertEqual(status, 200)
        self.assertTrue(kind.startswith("text/event-stream"))
        self.assertEqual(raw.decode(), "data: 0\n\ndata: 1\n\ndata: 2\n\ndata: [DONE]\n\n")

    async def test_upstream_down(self) -> None:
        await self.upstream.close()
        status, _, _ = await self.call("GET", "/v1/models", {"Authorization": f"Bearer {KEY}"})
        self.assertEqual(status, 502)


if __name__ == "__main__":
    unittest.main()
