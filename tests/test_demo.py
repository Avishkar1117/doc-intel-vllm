"""Public demo page: routes, limits and cost protection, with a fake model client and an
in-memory ledger - no GPU, no Azure, no network."""

import io
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel

import docintel.demo as demo_module
from docintel.api import app
from docintel.demo import (
    DemoSettings,
    Limiters,
    LocalSource,
    SampleStore,
    get_demo_settings,
    get_ledger,
    get_limiters,
    get_llm_client,
    get_sample_store,
)
from docintel.ledger import Ledger, LedgerConfig, MemoryStore
from docintel.schemas import Usage

GOOD_OUTPUT = json.dumps(
    {"company": "ACME TRADING", "address": "1 Test Road", "date": "01/02/2019", "total": 12.5}
)


def _sample(sample_id: str, company: str = "ACME TRADING") -> dict[str, Any]:
    return {
        "id": sample_id,
        "image": f"{sample_id}.jpg",
        "image_id": 1,
        "role": "Random pick; both systems answered.",
        "gold": {
            "company": company, "date": "01/02/2019", "address": "1 Test Road", "total": "12.50"
        },
        "self_hosted": {
            "company": company, "address": "1 Test Road", "date": "01/02/2019", "total": 12.5,
            "latency_ms": 3000, "fields_wrong": [],
        },
        "di": {
            "status": "no_total", "company": company, "address": "1 Test Road",
            "date": "2019-02-01", "total": None, "latency_ms": None, "fields_wrong": ["total"],
        },
    }  # fmt: skip


class FakeLLM:
    def __init__(self, outcome: str | Exception = GOOD_OUTPUT) -> None:
        self.outcome = outcome
        self.calls = 0

    def complete(
        self, image_bytes: bytes, prompt: str, schema: type[BaseModel]
    ) -> tuple[str, Usage]:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome, Usage(
            prompt_tokens=10, completion_tokens=5, visual_tokens=3, total_tokens=15
        )


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


class Env:
    def __init__(self, tmp_path: Path, **overrides: Any) -> None:
        samples = tmp_path / "samples"
        samples.mkdir()
        bundle = {
            "label": "Hand-picked to show different outcomes, not a representative sample.",
            "table": [
                {
                    "system": "self-hosted", "overall_f1": "0.914", "company_f1": "0.925",
                    "address_f1": "0.939", "date_f1": "0.965", "total_f1": "0.826",
                    "latency_p50_ms": "3893", "usd_per_1k_docs": "0.91",
                }
            ],
            "benchmark": {
                "n_receipts": 347, "di_no_total": 15, "di_wrong_negative_total": 4,
                "f1_both_answered": {"n": 326, "self_hosted": 0.914, "di": 0.871},
                "vllm_version": "0.21.0", "di_api_version": "2024-11-30",
            },
            "samples": [_sample("s01", "<script>alert(1)</script>"), _sample("s02")],
        }  # fmt: skip
        (samples / "samples.json").write_text(json.dumps(bundle), encoding="utf-8")
        for name in ("s01.jpg", "s02.jpg"):
            (samples / name).write_bytes(b"\xff\xd8fake-jpeg")
        (tmp_path / "secret.txt").write_text("do not serve")

        self.settings = DemoSettings(
            vllm_base_url="http://gpu.invalid", request_timeout_s=5, **overrides
        )
        self.clock = Clock()
        self.ledger = Ledger(
            MemoryStore(),
            LedgerConfig(
                budget_usd=self.settings.budget_usd,
                gpu_usd_per_s=0.001,
                margin=1.0,
                scaledown_window_s=100,
                cold_start_s=300,
                daily_request_cap=self.settings.daily_request_cap,
            ),
            self.clock,
        )
        self.llm = FakeLLM()
        self.store = SampleStore(LocalSource(samples))
        self.limiters = Limiters(
            extract=demo_module.SlidingWindowLimiter(self.settings.extract_per_ip_per_hour, 3600),
            warm=demo_module.SlidingWindowLimiter(self.settings.warm_per_ip_per_hour, 3600),
            page=demo_module.SlidingWindowLimiter(1000, 60),
            slots=threading.BoundedSemaphore(2),
        )

    def install(self, *, ledger: Any = "default", llm: Any = "default") -> TestClient:
        app.dependency_overrides[get_demo_settings] = lambda: self.settings
        chosen_ledger = self.ledger if ledger == "default" else ledger
        chosen_llm = self.llm if llm == "default" else llm
        app.dependency_overrides[get_ledger] = lambda: chosen_ledger
        app.dependency_overrides[get_llm_client] = lambda: chosen_llm
        app.dependency_overrides[get_sample_store] = lambda: self.store
        app.dependency_overrides[get_limiters] = lambda: self.limiters
        return TestClient(app)

    def wake(self) -> None:
        self.ledger.begin_warm()
        self.ledger.mark_ready()


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    yield Env(tmp_path)
    app.dependency_overrides.clear()


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client: TestClient, data: bytes, name: str = "r.png") -> httpx.Response:
    response: httpx.Response = client.post(
        "/demo/extract", files={"file": (name, data, "image/png")}
    )
    return response


def test_page_shows_samples_notice_and_security_headers(env: Env) -> None:
    response = env.install().get("/demo")
    assert response.status_code == 200
    assert "not a representative sample" in response.text
    assert "/demo/sample/s02.jpg" in response.text
    assert "not saved by this app" in response.text
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_model_and_gold_text_is_html_escaped(env: Env) -> None:
    response = env.install().get("/demo?sample=s01")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "<script>alert(1)" not in response.text


def test_sample_image_is_served_and_nothing_else(env: Env) -> None:
    client = env.install()
    assert client.get("/demo/sample/s01.jpg").status_code == 200
    assert client.get("/demo/sample/samples.json").status_code == 404
    assert client.get("/demo/sample/..%2Fsecret.txt").status_code == 404
    assert client.get("/demo/sample/secret.txt").status_code == 404


def test_upload_happy_path_charges_the_ledger(env: Env) -> None:
    client = env.install()
    env.wake()
    response = upload(client, png_bytes())
    assert response.status_code == 200
    assert "ACME TRADING" in response.text
    assert "12.50" in response.text
    assert env.llm.calls == 1
    assert env.ledger.snapshot().day_requests == 1


def test_upload_needs_an_awake_gpu(env: Env) -> None:
    response = upload(env.install(), png_bytes())
    assert response.status_code == 409
    assert "asleep" in response.text
    assert env.llm.calls == 0


def test_pdf_and_garbage_are_rejected_without_a_model_call(env: Env) -> None:
    client = env.install()
    env.wake()
    assert upload(client, b"%PDF-1.4 not an image", "a.pdf").status_code == 415
    assert upload(client, b"just text").status_code == 415
    assert env.llm.calls == 0


def test_oversize_upload_is_rejected_before_parsing(tmp_path: Path) -> None:
    env = Env(tmp_path, max_upload_bytes=500)
    try:
        client = env.install()
        env.wake()
        assert upload(client, b"x" * 5000).status_code == 413
        assert env.llm.calls == 0
    finally:
        app.dependency_overrides.clear()


def test_per_ip_hourly_limit(tmp_path: Path) -> None:
    env = Env(tmp_path, extract_per_ip_per_hour=1)
    try:
        client = env.install()
        env.wake()
        assert upload(client, png_bytes()).status_code == 200
        assert upload(client, png_bytes()).status_code == 429
        assert env.llm.calls == 1
    finally:
        app.dependency_overrides.clear()


def test_spent_budget_blocks_live_requests_but_not_samples(tmp_path: Path) -> None:
    env = Env(tmp_path, budget_usd=0.0)
    try:
        client = env.install()
        assert upload(client, png_bytes()).status_code == 503
        assert env.llm.calls == 0
        page = client.get("/demo")
        assert page.status_code == 200
        assert "budget" in page.text
        assert "/demo/sample/s02.jpg" in page.text
    finally:
        app.dependency_overrides.clear()


def test_unreadable_ledger_fails_closed(env: Env) -> None:
    class Broken:
        def load(self) -> tuple[None, None]:
            raise OSError("down")

        def save(self, data: dict[str, Any], token: str | None) -> None:
            raise OSError("down")

    broken = Ledger(Broken(), env.ledger._config, env.clock)
    client = env.install(ledger=broken)
    assert upload(client, png_bytes()).status_code == 503
    assert env.llm.calls == 0
    assert client.get("/demo").status_code == 200  # samples still work


def test_live_is_off_when_no_gpu_is_configured(env: Env) -> None:
    client = env.install(llm=None)
    assert upload(client, png_bytes()).status_code == 503
    assert client.post("/demo/warm").status_code == 503


def test_upstream_failure_marks_the_gpu_cold(env: Env) -> None:
    env.llm.outcome = httpx.ConnectError("gone")
    client = env.install()
    env.wake()
    assert upload(client, png_bytes()).status_code == 503
    assert env.ledger.snapshot().gpu == "cold"


def test_unreadable_model_output_shows_no_result_but_is_charged(env: Env) -> None:
    env.llm.outcome = "not json at all"
    client = env.install()
    env.wake()
    response = upload(client, png_bytes())
    assert response.status_code == 200
    assert "could not find a readable total" in response.text
    assert env.ledger.snapshot().day_requests == 1


def test_warm_pokes_the_gpu_once(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    pokes: list[int] = []
    monkeypatch.setattr(demo_module, "_poke_gpu", lambda settings: pokes.append(1))
    client = env.install()
    assert client.post("/demo/warm").json() == {"outcome": "started"}
    assert client.post("/demo/warm").json() == {"outcome": "already"}
    threading.Event().wait(0.1)  # let the daemon thread run
    assert pokes == [1]


def test_warm_is_rate_limited_per_ip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo_module, "_poke_gpu", lambda settings: None)
    env = Env(tmp_path, warm_per_ip_per_hour=1)
    try:
        client = env.install()
        assert client.post("/demo/warm").status_code == 200
        assert client.post("/demo/warm").status_code == 429
    finally:
        app.dependency_overrides.clear()


def test_status_never_probes_a_sleeping_gpu(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(settings: DemoSettings) -> bool:
        raise AssertionError("status must not touch a sleeping GPU")

    monkeypatch.setattr(demo_module, "_gpu_healthy", fail)
    assert env.install().get("/demo/status").json()["gpu"] == "cold"


def test_status_flips_to_ready_once_the_warming_gpu_answers(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo_module, "_gpu_healthy", lambda settings: True)
    monkeypatch.setattr(demo_module, "_poke_gpu", lambda settings: None)
    client = env.install()
    client.post("/demo/warm")
    assert client.get("/demo/status").json()["gpu"] == "ready"
