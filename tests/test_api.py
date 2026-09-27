"""Mocked-vLLM integration test for api.py - exercises the real FastAPI app, HTTP layer
included, but against MockClient (config.py's default backend), so it runs on any machine
with no GPU and no live Modal endpoint. Per CLAUDE.md: tests run without a GPU.
"""

from fastapi.testclient import TestClient

from docintel.api import app

client = TestClient(app)


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
