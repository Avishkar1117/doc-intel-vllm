"""vLLM server for the public demo page, deployed in its own Modal environment (`demo`).

Same image and flags as modal_app.py's 4B reduced-resolution run, so a live upload behaves
like the ten cached SROIE samples beside it. Separate app and environment so the demo can
never share a container, a budget or a URL with the benchmark server.
"""

import subprocess

import modal

MINUTES = 60
VLLM_PORT = 8000
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
# The exact config the cached SROIE answers were captured with (Phase 8: 4B, reduced res).
MM_KWARGS = '{"size": {"longest_edge": 1500000, "shortest_edge": 65536}}'

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Weights were downloaded once in the `main` environment; mounting that Volume avoids a
# second ~8 GB download. The compile cache gets its own Volume so the two apps never write
# into the same one.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", environment_name="main")
vllm_cache_vol = modal.Volume.from_name("vllm-cache-demo", create_if_missing=True)

app = modal.App("vllm-doc-intelligence-demo")


@app.server(
    image=vllm_image,
    gpu="L4",
    port=VLLM_PORT,
    startup_timeout=MINUTES * 10,  # above the worst measured cold start (6m24s)
    unauthenticated=False,  # every caller needs a Modal proxy auth token
    # Idle time before the container shuts down. The app's spend ledger prices exactly this
    # window (DOCINTEL_DEMO_SCALEDOWN_WINDOW_S), so the two numbers must stay equal.
    scaledown_window=MINUTES * 5,
    # One GPU is the hard cost ceiling: nobody needs two at once for a demo, and the
    # ledger's cost model assumes a single container.
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class Server:
    @modal.enter()
    def start(self) -> None:
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
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.9",
            # Same value as every measured config; the app only admits 2 live requests at
            # once, so the extra slots are headroom, not memory that has to be paid for.
            "--max-num-seqs",
            "8",
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
