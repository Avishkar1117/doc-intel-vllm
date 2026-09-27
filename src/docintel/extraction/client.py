"""OpenAI-compatible client - the seam that swaps between a mocked local client and a
real HTTP backend (Modal, ephemeral or deployed) via one env var.

Never talks to Azure directly: Azure only ever hosts the FastAPI app and Document
Intelligence (baseline.py) - neither one serves the model (PROJECT_BRIEF.md §4). The
same HTTPClient works for both an ephemeral `modal run` job and the persistent
`modal deploy` endpoint, since both speak the same OpenAI-compatible wire protocol -
only the base_url differs.
"""

import base64
from typing import Protocol

import httpx
import structlog
from pydantic import BaseModel

from docintel.config import settings
from docintel.schemas import LineItem, Receipt, Subtotals, Totals, Usage

logger = structlog.get_logger(__name__)


class VLLMClient(Protocol):
    """What pipeline.py needs from any backend: image+prompt in, JSON text + usage out."""

    def complete(
        self, image_bytes: bytes, prompt: str, schema: type[BaseModel]
    ) -> tuple[str, Usage]: ...


# A fixed, schema-valid example - not derived from `schema` generically, since the mock
# only ever stands in for Receipt in this project. GTX 1650 can't run Qwen3-VL's vision
# tower at all (D-001), so there is no "real" local output to approximate; this just
# lets the pipeline and its tests run end to end with zero GPU cost.
_MOCK_RECEIPT = Receipt(
    line_items=[
        LineItem(
            name="Nasi Goreng",
            count=1.0,
            unit_price=25000.0,
            price=25000.0,
            discount_price=None,
        ),
    ],
    subtotals=Subtotals(
        subtotal=25000.0,
        discount=None,
        service_charge=None,
        other_service_charge=None,
        tax=2500.0,
    ),
    totals=Totals(
        total=27500.0,
        cash_paid=30000.0,
        change_due=2500.0,
        credit_card_paid=None,
        emoney_paid=None,
        menu_type_count=None,
        menu_quantity_count=None,
    ),
)


class MockClient:
    """Local-dev backend - per CLAUDE.md, the only one the local machine may ever run."""

    def complete(
        self, image_bytes: bytes, prompt: str, schema: type[BaseModel]
    ) -> tuple[str, Usage]:
        raw_output = _MOCK_RECEIPT.model_dump_json()
        # Rough char/4 estimate, not a real tokenizer - good enough for exercising the
        # pipeline, not meant to produce a trustworthy cost number.
        prompt_tokens = max(len(prompt) // 4, 1)
        completion_tokens = max(len(raw_output) // 4, 1)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            visual_tokens=0,  # mock never processes an actual image
            total_tokens=prompt_tokens + completion_tokens,
        )
        return raw_output, usage


class HTTPClient:
    """Talks to any real OpenAI-compatible vLLM endpoint - a Modal server, ephemeral or deployed."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)
        self._text_only_cache: dict[str, int] = {}

    def complete(
        self, image_bytes: bytes, prompt: str, schema: type[BaseModel]
    ) -> tuple[str, Usage]:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        text_block: dict[str, object] = {"type": "text", "text": prompt}
        payload = self._payload(
            [
                text_block,
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
            schema,
        )
        response = self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()

        raw_output: str = body["choices"][0]["message"]["content"]
        usage_body = body["usage"]
        logger.debug(
            "vllm_completion",
            prompt_tokens=usage_body["prompt_tokens"],
            completion_tokens=usage_body["completion_tokens"],
        )

        # vLLM 0.21.0 reports no per-request image-token breakdown, so visual tokens are
        # the prompt_tokens of this request minus the same prompt sent text-only. Verified
        # in the Phase 6 spike: 5816 measured vs 5814 recorded in Phase 2 (the 2-token gap
        # is the vision start/end placeholders, so this slightly overstates image tokens).
        visual_tokens = usage_body["prompt_tokens"] - self._text_only_tokens(prompt, schema)
        usage = Usage(
            prompt_tokens=usage_body["prompt_tokens"],
            completion_tokens=usage_body["completion_tokens"],
            visual_tokens=visual_tokens,
            total_tokens=usage_body["total_tokens"],
        )
        return raw_output, usage

    def _payload(
        self, content: list[dict[str, object]], schema: type[BaseModel]
    ) -> dict[str, object]:
        """One body builder, so the image and text-only calls can't drift apart."""
        return {
            "model": settings.model_name,
            "messages": [{"role": "user", "content": content}],
            # response_format, not the removed guided_json (D-008) - confirmed against
            # the v0.21.0 docs, this is vLLM's OpenAI-standard structured-output path.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
        }

    def _text_only_tokens(self, prompt: str, schema: type[BaseModel]) -> int:
        """Prompt tokens with no image; cached since the prompt is constant across receipts."""
        if prompt not in self._text_only_cache:
            payload = self._payload([{"type": "text", "text": prompt}], schema)
            payload["max_tokens"] = 1  # accounting only, no real generation
            payload["temperature"] = 0
            response = self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            self._text_only_cache[prompt] = int(response.json()["usage"]["prompt_tokens"])
        return self._text_only_cache[prompt]


def get_client() -> VLLMClient:
    """Reads DOCINTEL_BACKEND and returns the matching client - callers never branch on backend."""
    if settings.backend == "mock":
        return MockClient()
    return HTTPClient(settings.vllm_base_url, settings.vllm_api_key)
