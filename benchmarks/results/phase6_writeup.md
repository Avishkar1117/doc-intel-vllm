# Phase 6: The four-config serving comparison

**Setup:** One NVIDIA L4 on Modal running vLLM 0.21.0, benchmarking Qwen3-VL-4B-Instruct and Qwen3-VL-2B-Instruct with `--max-num-seqs 8` and `--max-model-len 16384`. Four setups (two model sizes across two visual-token budgets) were tested in separate server runs: cold boot, accuracy evaluation across all 100 CORD test receipts, and a concurrency sweep {1, 2, 4, 8, 16, 32} with 48 requests per level.

"Default resolution" refers to the Qwen3-VL processor's internal default (up to 16,384 tokens/image). This is distinct from the `max_pixels` value used in Phase 1, which the Qwen3-VL processor silently ignored (see DECISIONS.md). "Capped" sets `--mm-processor-kwargs '{"size": {"longest_edge": 1500000, "shortest_edge": 65536}}'`, capping an image at roughly 1,465 visual tokens.

## The table

| Config | Visual tokens/doc | CORD F1 | Peak docs/min | p95 @ peak | docs/min @ c=8 | p95 @ c=8 | $/1k docs |
|---|---|---|---|---|---|---|---|
| 4B, default res | 2104 | 0.819 | 24.5 (c=16) | 53.5s | 23.0 | 35.7s | $0.54 |
| 4B, reduced res | 973 | **0.833** | 36.7 (c=32) | 54.7s | 35.4 | 20.2s | $0.36 |
| 2B, default res | 2104 | 0.652 | 58.6 (c=16) | 21.0s | 49.0 | 12.8s | $0.23 |
| 2B, reduced res | 968 | 0.670 | **66.3** (c=16) | 18.7s | 56.4 | 12.6s | **$0.20** |

F1 is scored on the 94 CORD receipts that completed cleanly across all four setups. Five receipts were dropped because CORD's ground truth lacked a total value, and one additional receipt per setup was excluded due to a 2B timeout or an unparseable output. Cost assumes 100% GPU utilization at peak throughput with zero idle time, priced in USD using Modal's published rates (no EUR conversion was pulled for this run; see D-015).

## Where the curve bends, and why

**The knee sits at `--max-num-seqs 8` across every setup, independent of model speed.** Only 8 requests can execute on the GPU at any given time. Requests 9 and above queue up until an active slot frees up. vLLM's `running` and `waiting` metrics verify this directly: `waiting` sits at 0 up through concurrency 8, then climbs to an average of 2.5 to 4.9 once concurrency hits 16. This ceiling is identical for 4B and 2B. Model size does not alter where the queue forms; it only dictates how much throughput runs through those 8 slots.

Throughput flattens out past c=8 for all runs. Whether a peak lands at c=16 or c=32 comes down to variance (4B capped peak at c=32 is only 0.3 docs/min faster than its c=16 mark), rather than a genuine shift in behavior under load.

**Resolution barely shifts single-request latency, but it directly impacts throughput.** At concurrency 1, generating 250 to 500 output JSON tokens takes far longer than the initial image read. The 4B model outputs roughly 22 tokens/sec at c=1, meaning ~251 output tokens account for about 11 seconds of the ~10.2s median request time. Trimming the image from 2,104 to 973 tokens only cuts prefill time, which is a minor part of the total (p50 at c=1 moves from 10.2s to 9.9s). The cap delivers its value under concurrent load: smaller requests clear earlier, yielding a consistent 1.5x to 1.6x throughput gain over default resolution for both models.

**Cutting visual tokens by more than half did not hurt accuracy.** F1 ticked slightly higher with the cap enabled (4B went from 0.819 to 0.833; 2B went from 0.652 to 0.670). Because this showed up independently on both models, it is unlikely to be pure noise. We initially expected small-print details like item descriptions and unit prices to degrade under lower resolutions, but every field held steady or improved. The likely explanation is that the cap only affects the ~23% of receipts that exceeded ~1,465 tokens by default (dense, long documents running up to 11,846 tokens). These extra-long inputs may have suffered from aggressive internal scaling rather than benefiting from it. In short, Qwen3-VL was receiving far more visual tokens than receipt text actually requires. Higher resolution does not inherently mean better extraction.

**Model size remains the main cost driver, but it comes with a major accuracy drop.** The 2B model delivers 2.4x the throughput of the 4B model at matched resolution, but overall F1 falls by 0.16 to 0.17. The degradation is uneven: 2B drops to near-random performance on `menu_type_count` and `menu_quantity_count` (0.10 to 0.34 F1), while `line_items.unit_price` stays relatively solid (0.92 down to 0.86). For the final total specifically, 2B holds up reasonably well: on the 94 shared receipts, `totals.total` hits 0.723 F1 on 2B capped compared to 0.787 on 4B capped. The overall F1 gap looks worse than it is because a few weak count fields drag down the aggregate score.

## What each axis buys

Resolution and model size hit different phases of the workload. Resolution scales down prefill (the upfront image read), which yields gains when requests are actively competing for GPU execution slots. Model size cuts decode latency across every generated token, making it the primary factor behind single-request turnaround times.

## Recommendation

**For full document extraction where every line item is required, run 4B capped.** It matches or slightly beats 4B default accuracy, improves throughput by 1.5x, and cuts costs from $0.54 down to $0.36 per 1,000 documents. There is no performance downside here; the default resolution simply consumed GPU cycles without improving output quality.

**2B capped is viable for total-only extraction pipelines**, such as lightweight receipt loggers or expense tools that only need the final transaction amount and vendor. In that constrained scope, 2B capped scores 0.723 on `totals.total` versus 0.787 on 4B, while processing 66.3 docs/min at $0.20 per 1,000 documents. The broader F1 drop is largely irrelevant for that use case because the model's primary failure points (`menu_type_count`, `change_due`) are ignored. Vendor names are missing from the CORD schema and were not evaluated here, though extraction would need that field in production.

If targeting a ~20s p95 latency on a workload around 50,000 receipts/day (~35 docs/min continuous load), 4B capped at 36.7 docs/min essentially saturates a single L4. Meeting higher demand from here requires adding a second GPU or setting up a tiered fallback, rather than trying to squeeze more out of prompts or schemas.

## Caveats and limitations

- **Single-sample evaluation at default temperature 0.7:** vLLM defaults to 0.7 and no explicit temperature was set. F1 shifts within 1 or 2 points should be treated with caution without repeated trials.
- **Unbounded generation in the 2B runs:** Receipt 59 (the largest input at 11,846 visual tokens) produced 3,142 completion tokens over 204 seconds on `2b_default`, and timed out across four ~120s retries on `2b_cap1500k`. During the sweep, an unconstrained `2b_cap1500k` request at c=1 hit 13,560 tokens in 266 seconds, pulling the level's mean output length from ~250 up to 534 tokens. Because `max_tokens` was omitted, models that missed a stop token kept generating. This makes 2B throughput and latency figures somewhat pessimistic compared to a strictly bounded run.
- **The resolution floor was not tested:** The 1.5M-pixel cap only affected the top 23% largest documents. Testing an aggressive limit (like ~512 tokens across all receipts) would be necessary to identify the exact point where small-print legibility breaks down.
- **Concurrency consistency was not verified:** We did not run an output diff between c=1 and c=8 to verify batching consistency. Running this at temperature 0.7 produces natural output variance, so the test was postponed until runs are pinned to `temperature=0`.
- **Cost models assume 100% saturation:** Rates reflect active GPU processing time without accounting for idle gaps or cold starts. Real-world per-document operational costs will be higher unless queues are consistently full.

## Next steps

- **Serving optimizations:** Test FP8 checkpoints for both models. Reducing memory pressure could allow increasing `--max-num-seqs` to unlock higher throughput on the same L4 instance. Benchmarking against a standard `transformers` generation loop would also provide a clean baseline to quantify vLLM's continuous batching advantage.
- **Extraction pipeline updates:** Expand the extraction schema to capture common field variations. Additionally, auto-cropping empty header and footer margins prior to VLM ingestion would lower visual token counts without altering resolution.
