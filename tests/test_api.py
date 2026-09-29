"""Mocked-vLLM integration test for api.py - exercises the real FastAPI app, HTTP layer
included, but against MockClient (config.py's default backend), so it runs on any machine
with no GPU and no live Modal endpoint. Per CLAUDE.md: tests run without a GPU.
"""

from fastapi.testclient import TestClient
from starlette.requests import Request

from docintel.api import app
from docintel.netutil import client_ip as _client_ip

client = TestClient(app)


def _request_with_headers(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request({"type": "http", "headers": headers, "client": ("10.0.0.1", 1234)})


def test_client_ip_uses_rightmost_forwarded_entry() -> None:
    # A caller-supplied leading value must not decide the rate-limit bucket: the ingress
    # appends the real IP, so only the rightmost entry is trustworthy.
    spoofed = _request_with_headers([(b"x-forwarded-for", b"203.0.113.77, 198.51.100.9")])
    assert _client_ip(spoofed) == "198.51.100.9"


def test_client_ip_falls_back_to_peer_address() -> None:
    assert _client_ip(_request_with_headers([])) == "10.0.0.1"


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_returns_valid_envelope() -> None:
    # Content doesn't matter to MockClient - it never looks at the bytes, only pipeline.py's
    # prompt/schema plumbing runs for real here.
    files = {"file": ("receipt.jpg", b"not a real image", "image/jpeg")}
    response = client.post("/extract", files=files)

    assert response.status_code == 200
    body = response.json()

    # MockClient's fixed example always passes business rules first-pass (client.py) -
    # a real regression here (a broken schema field, a validate.py off-by-one) shows up
    # as this flipping to False without a single live GPU request.
    assert body["validation"]["passed"] is True
    assert body["validation"]["repair_attempts"] == 0
    assert body["data"]["totals"]["total"] == 27500.0

    # cost.py's contribution: present, non-negative, derived from this response's own latency.
    assert body["estimated_cost_usd"] >= 0
