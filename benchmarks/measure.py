import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from docintel.eval.capture import VLLM_VERSION

RESULTS = Path("benchmarks/results")
MANIFEST = RESULTS / "phase6_workload.json"
APP_NAME = "vllm-doc-intelligence"

# tokens ~ longest_edge / 1024 (measured: 254 at 262144); 1.5M caps a receipt at ~1465 tokens
CAP_1500K = '{"size": {"longest_edge": 1500000, "shortest_edge": 65536}}'

REQUESTS_PER_LEVEL: int = 48 # same as phase 5 baseline

L4_USD_PER_S = 0.000222  # PROJECT_BRIEF.md §3


@dataclass(frozen=True)
class Config:
    tag: str
    model: str
    mm_kwargs: str


CONFIGS = [
    Config("4b_default", "Qwen/Qwen3-VL-4B-Instruct", "{}"),
    Config("4b_cap1500k", "Qwen/Qwen3-VL-4B-Instruct", CAP_1500K),
    Config("2b_default", "Qwen/Qwen3-VL-2B-Instruct", "{}"),
    Config("2b_cap1500k", "Qwen/Qwen3-VL-2B-Instruct", CAP_1500K),
]


def config_env(cfg: Config) -> dict[str, str]:
    return {
        **os.environ,
        "DOCINTEL_MODEL": cfg.model,
        "DOCINTEL_MM_KWARGS": cfg.mm_kwargs,
        # request "model" must equal --served-model-name, or vLLM answers 404
        "DOCINTEL_MODEL_NAME": cfg.model,
        "PYTHONIOENCODING": "utf-8",  # Modal's checkmarks crash the Windows charmap codec (Phase 1)
    }


def stop_app() -> None:
    # check=False: stopping an app that isn't running is not an error worth aborting over
    subprocess.run([sys.executable, "-m", "modal", "app", "stop", APP_NAME, "-y"], check=False)


def mean_visual_tokens(capture_path: Path) -> float:
    """Mean visual tokens per accepted receipt - the cost variable that goes in the table."""
    counts: list[int] = []
    with capture_path.open() as f:
        for line in f:
            prediction = json.loads(line)["prediction"]
            if prediction is not None:
                counts.append(prediction["usage"]["visual_tokens"])
    return sum(counts) / len(counts)


def run_config(cfg: Config, base_url: str) -> None:
    env = config_env(cfg)
    capture_path = RESULTS / f"phase6_{cfg.tag}_capture.jsonl"
    summary_path = RESULTS / f"phase6_{cfg.tag}_summary.json"
    if summary_path.exists():
        print(f"{cfg.tag}: already finished")
        return

    stop_app()
    try:
        subprocess.run(
            [sys.executable, "-m", "modal", "deploy", "modal_app.py"], env=env, check=True
        )
        subprocess.run(
            [sys.executable, "-m", "docintel.eval.capture",
             "--base-url", base_url, "--output", str(capture_path)],
            env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "benchmarks/benchmark.py", "run", "--base-url", base_url,
             "--manifest", str(MANIFEST), "--tag", f"phase6_{cfg.tag}",
             "--requests-per-level", str(REQUESTS_PER_LEVEL)],
            env=env, check=True,
        )
    finally:
        stop_app()  # never leave a GPU idling after a crash (Phase 5 gotcha)

    summary = {
        "tag": cfg.tag, "model": cfg.model, "mm_processor_kwargs": cfg.mm_kwargs,
        "vllm_version": VLLM_VERSION, "mean_visual_tokens": mean_visual_tokens(capture_path),
    }
    (RESULTS / f"phase6_{cfg.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{cfg.tag}: done, mean visual tokens {summary['mean_visual_tokens']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="the Modal server URL (same for every config)"
    )
    parser.add_argument("--config", choices=[c.tag for c in CONFIGS] + ["all"], required=True)
    args = parser.parse_args()

    selected = CONFIGS if args.config == "all" else [c for c in CONFIGS if c.tag == args.config]
    # rough budget: ~40 min of L4 per config (boot + capture + sweep) - measured actual is
    # higher (see DECISIONS.md, this phase - the duplicated timed-batch call in benchmark.py)
    est_usd = len(selected) * 40 * 60 * L4_USD_PER_S
    prompt = f"{len(selected)} config(s), est. ~${est_usd:.2f} of GPU. Proceed? [y/N] "
    if input(prompt).lower() != "y":
        return

    for cfg in selected:
        try:
            run_config(cfg, args.base_url)
        except subprocess.CalledProcessError:
            print(f"{cfg.tag}: failed, skipping to next config")
            continue


if __name__ == "__main__":
    main()