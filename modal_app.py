import base64
import os
import pathlib

import httpx
import modal

# image = base cuda devel img + vllm via uv 

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )

MODEL_NAME = os.environ.get("DOCINTEL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
MM_KWARGS = os.environ.get("DOCINTEL_MM_KWARGS", "{}")
MAX_NUM_SEQS = os.environ.get("DOCINTEL_MAX_NUM_SEQS", "8")
# weights cache
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("vllm-doc-intelligence")

MINUTES = 60
VLLM_PORT = 8000

@app.server(
    image=vllm_image,
    gpu="L4",
    scaledown_window=MINUTES * 15, # scale down after 15 minutes of inactivity
    max_containers = 1,
    # first boot pulls weights into the Volume; generous ceiling for that
    startup_timeout=MINUTES * 10,
    port=VLLM_PORT,
    unauthenticated=True,  # allow public access to the server
    env={
        "DOCINTEL_MODEL": MODEL_NAME,
        "DOCINTEL_MM_KWARGS": MM_KWARGS,
        "DOCINTEL_MAX_NUM_SEQS": MAX_NUM_SEQS,  
    },
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    }
)

class Server:
    @modal.enter()
    def start(self) -> None:
        import subprocess
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=info",
        ]

        cmd += [
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.9",
            "--max-num-seqs",
            MAX_NUM_SEQS,
            "--mm-processor-kwargs",
            MM_KWARGS,
            "--mm-processor-cache-gb",
            "0",
        ]

        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()

@app.local_entrypoint()
def test(image_path: str, health_timeout: int = 10 * 60) -> None:
    """Phase 1 DoD smoke test: one receipt image in, print whatever comes back.

    Invoke: modal run modal_app.py --image-path path/to/receipt.jpg
    """
    import time

    url = Server.get_url()

    # Cold start pulls the image, installs vLLM, and downloads ~8.5GB of weights —
    # poll /health instead of firing the real request immediately, or the ephemeral
    # `modal run` app tears itself down on the first 503 before the container
    # finishes booting.
    print(f"Waiting for {url} to become healthy (cold start can take several minutes)...")
    deadline = time.time() + health_timeout
    healthy = False
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=10).status_code == 200:
                healthy = True
                break
        except httpx.RequestError:
            pass
        time.sleep(5)
    if not healthy:
        raise TimeoutError(
            f"Server never became healthy within {health_timeout}s — check the container logs."
        )
    print("Server healthy, sending request.")

    image64 = base64.b64encode(pathlib.Path(image_path).read_bytes()).decode()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "extract all the visible text and structured fields "
                            "from this receipts as JSON."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image64}"},
                    },
                ],
            }
        ],
    }

    response = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=600)
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])  