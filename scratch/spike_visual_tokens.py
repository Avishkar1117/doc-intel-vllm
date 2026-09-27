"""Phase 6 spike: does a server-level mm-processor kwarg actually change visual-token count?

Sends one receipt twice - with the image, and text-only with the identical prompt - and
reports the prompt_tokens difference. Run against each candidate server launch and compare
the number to the 5814 baseline (test receipt from Phase 2). Throwaway, like the other
scratch/ scripts.
"""

import argparse
import base64
import sys
import time
from pathlib import Path

import httpx

PROMPT = "Extract all visible text and fields from this receipt as JSON."


def wait_healthy(client: httpx.Client, timeout_s: int) -> None:
    # Cold start is ~6 min; poll for several consecutive 200s because a stale replica can
    # answer /health while the new one is still booting (Phase 2 gotcha).
    deadline = time.time() + timeout_s
    ok = 0
    while time.time() < deadline:
        try:
            ok = ok + 1 if client.get("/health", timeout=10).status_code == 200 else 0
        except httpx.RequestError:
            ok = 0
        if ok >= 3:
            return
        time.sleep(5)
    sys.exit(f"server not healthy within {timeout_s}s")


def prompt_tokens(client: httpx.Client, model: str, content: list[dict[str, object]]) -> int:
    # max_tokens=1: we only want the server's prompt accounting, not a generation.
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    resp.raise_for_status()
    return int(resp.json()["usage"]["prompt_tokens"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--health-timeout", type=int, default=600)
    args = ap.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        wait_healthy(client, args.health_timeout)
        b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")
        text = {"type": "text", "text": PROMPT}
        image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        with_image = prompt_tokens(client, args.model, [text, image])
        text_only = prompt_tokens(client, args.model, [text])

    # The difference also includes a few vision-start/end placeholder tokens from the chat
    # template - a small constant, so it slightly overstates pure image tokens.
    print(f"model={args.model}")
    print(f"prompt_tokens with image : {with_image}")
    print(f"prompt_tokens text-only  : {text_only}")
    print(f"visual tokens (approx)   : {with_image - text_only}")


if __name__ == "__main__":
    main()
