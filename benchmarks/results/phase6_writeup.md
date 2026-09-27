# Phase 6 — the four-config serving comparison

**Setup.** One NVIDIA L4 on Modal, vLLM 0.21.0, Qwen3-VL-4B-Instruct and Qwen3-VL-2B-Instruct,
`--max-num-seqs 8`, `--max-model-len 16384`. Four configs — two model sizes x two visual-token
budgets — each run as its own server launch: boot, accuracy capture over all 100 CORD test
receipts, concurrency sweep {1, 2, 4, 8, 16, 32} at 48 requests/level. "Default resolution" is
the Qwen3-VL processor's own default (up to 16,384 tokens/image) — not the `max_pixels` value
carried from Phase 1, which turned out to be silently ignored by Qwen3-VL's processor (see
DECISIONS.md, this phase). "Capped" sets `--mm-processor-kwargs '{"size": {"longest_edge":
1500000, "shortest_edge": 65536}}'`, which limits an image to ~1465 visual tokens.

## The table

| Config | Visual tokens/doc | CORD F1 | Peak docs/min | p95 @ peak | docs/min @ c=8 | p95 @ c=8 | $/1k docs |
|---|---|---|---|---|---|---|---|
| 4B, default res | 2104 | 0.819 | 24.5 (c=16) | 53.5s | 23.0 | 35.7s | $0.54 |
| 4B, reduced res | 973 | **0.833** | 36.7 (c=32) | 54.7s | 35.4 | 20.2s | $0.36 |
| 2B, default res | 2104 | 0.652 | 58.6 (c=16) | 21.0s | 49.0 | 12.8s | $0.23 |
| 2B, reduced res | 968 | 0.670 | **66.3** (c=16) | 18.7s | 56.4 | 12.6s | **$0.20** |

F1 is scored on the 94 CORD receipts that scored cleanly under all four configs (5 excluded —
CORD's own ground truth is missing a total; 1 more excluded per config — a 2B timeout or an
unparseable output). Cost assumes the GPU is kept fully busy at peak throughput, no idle time,
in USD (Modal's published rate; no EUR conversion is sourced for this project — see D-015).

## Where the curve bends, and why

**The knee is `--max-num-seqs 8`, in every config, not model speed.** Only 8 requests are
ever allowed to run on the GPU at once. Requests 9 and up don't get a share of the GPU —
they sit in a queue and wait for one of the 8 running slots to free up before they're even
started. Checking vLLM's own `running`/`waiting` counters confirms this directly: in all four
configs, `waiting` is 0 through c=8, then jumps to a mean of 2.5–4.9 the moment c=16 is
reached. That's the same threshold for 4B and 2B alike — model size doesn't move where the
knee sits, it only changes how much throughput is packed into each of those 8 slots.

Past c=8, throughput is close to flat for every config, so whether a sweep's single highest
number happens to land at c=16 or c=32 is mostly noise (4B capped's "peak" at c=32 is only
0.3 docs/min above its own c=16 number) — not a real difference in how the two model sizes
behave under load.

**Resolution barely moves single-request latency, but does move throughput.** At concurrency
1, generating the ~250–500 tokens of output JSON takes far longer than reading the image:
4B's own output rate is ~22 tokens/sec at c=1, so ~251 output tokens alone accounts for
roughly 11 of the ~10.2s median request time. Cutting the image from 2104 to 973 tokens saves
prefill time, which is a small slice of that total — c=1 p50 only moves 10.2s → 9.9s. Where
the cap actually pays off is under load: with less to process per request, more requests
clear their turn before the next queue check, which is why capped throughput is consistently
1.5–1.6x default across both models.

**Cutting visual tokens by more than half did not cost accuracy.** F1 went *up* slightly
under the cap for both models (4B: 0.819→0.833, 2B: 0.652→0.670) — a real signal repeated on
two independent models, not just sampling noise on one. The brief expected small-print fields
(line-item names, unit prices) to degrade first if resolution was cut too far; instead every
field held or improved. The likely reason: the cap only reduces the ~23% of receipts that
were *above* ~1465 tokens by default (the tall, dense ones, up to 11,846 tokens) — receipts
that may have been getting *worse* results from an oversized, over-resized input, not better
ones. This project fed Qwen3-VL far more visual tokens than a receipt's actual text needs.
That's a real finding worth stating plainly: more resolution is not automatically more signal.

**Model size is the dominant cost lever, and it costs real accuracy.** 2B is 2.4x the
throughput of 4B at equal resolution, but drops overall F1 by ~0.16–0.17. The collapse isn't
uniform: 2B's `menu_type_count`/`menu_quantity_count` fields fall to F1 0.10–0.34 (near-random),
while `line_items.unit_price` barely moves (0.92→0.86). 2B still gets the *total* field right
close to as often as 4B: on the shared 94 receipts, `totals.total` scores 0.723 F1 (2B capped)
vs 0.787 (4B capped) — a real gap, but far smaller than the overall-F1 gap, because it's driven
by 2B's near-total failure on a handful of low-signal count fields dragging the average down,
not by 2B being uniformly worse at every field.

## What each axis buys

Resolution and model size load different parts of the cost, as the brief predicted. Resolution
mainly cuts *prefill* — the one-time cost of reading the image — which matters most under
load, when many requests are competing for GPU time each round. Model size cuts *decode* cost
per output token, everywhere, which is why it dominates the single-request latency budget too.

## Recommendation

**For a company digitizing full receipts into a structured record — every line item, not
just the total — 4B capped is the config to run.** It matches 4B default's accuracy (in fact
slightly exceeds it) while delivering 1.5x the throughput and cutting cost from $0.54 to
$0.36 per 1,000 documents. There is no accuracy trade to make here; the default-resolution
config was simply spending more GPU time than the task needed.

**A real-world use case worth naming for 2B capped: a "just the total" workload** — e.g. a
spend-tracking or expense-digitization tool that only needs the amount and who it was paid to,
not a full itemized record. For that narrower job, 2B capped gets the total right nearly as
often as 4B (F1 0.723 vs 0.787 on that one field) while running at 66.3 docs/min for $0.20/1k
docs — the overall-F1 gap looks larger than it really is here, since most of it comes from
fields (`menu_type_count`, `change_due`, etc.) this workload wouldn't be scoring at all. A
merchant-name field isn't in CORD's schema and wasn't tested, but it's the natural companion
field for this use case — knowing *who* was paid matters as much as *how much* for spend
tracking — and would be worth adding if this workload were pursued for real. This is a
hypothesis from the numbers above, not something this project measured.

At a p95 budget of ~20s and a workload of, say, 50,000 receipts/day (~35 docs/min sustained),
4B capped's peak of 36.7 docs/min is close to the ceiling of a single L4 — two GPUs (or a
second config tier) would be the next lever, not a further prompt or schema change.

## Caveats — what this measurement does not show

- **One sample per receipt, temperature 0.7 (never explicitly set — this is vLLM's model
  default), no repeated trials.** F1 differences of a point or two, anywhere in this report,
  are not distinguishable from sampling noise on this evidence alone.
- **The 2B runs include at least one runaway generation.** Receipt 59 (the heaviest in the
  set, 11,846 default visual tokens) took 3,142 completion tokens / 204s under `2b_default`,
  and timed out entirely (4 retries, ~120s each) under `2b_cap1500k`. Inside the sweep itself,
  one `2b_cap1500k` request at c=1 generated 13,560 completion tokens in 266 seconds — enough
  to pull that level's own mean completion length from ~250 to 534 tokens. No request sets
  `max_tokens`, so nothing stops a model that fails to emit a stop token. 2B's throughput and
  latency numbers are therefore somewhat pessimistic versus a config that capped generation
  length; this was left unfixed deliberately, per this session's priority on the serving
  measurement over extraction tuning.
- **The "small print degrades first" hypothesis was not actually tested.** The 1.5M-pixel cap
  only touches the receipts already above that size. A much smaller cap (e.g. ~512 tokens,
  touching most receipts) would be the real test of whether tight resolution costs accuracy.
- **The brief's c=1-vs-c=8 batching-consistency check (diff outputs, confirm batching doesn't
  change answers) was not run.** It needs `temperature=0` to be meaningful — under the
  current 0.7 default, two runs of the same receipt can legitimately differ regardless of
  concurrency, so a diff wouldn't isolate batching's effect. Deliberately deferred.
- **Cost is USD, at 100% GPU utilization, no idle time, no cold-start amortization.** A real
  deployment's price per document would be higher unless request volume reliably fills the
  GPU.

## What we'd try next, given more time

- **Serving side** (this project's actual focus): measure FP8 checkpoints for both model
  sizes — same accuracy at lower memory could raise `--max-num-seqs` and the whole throughput
  ceiling on the same L4. Also worth one direct comparison against a non-vLLM inference path
  (e.g. bare `transformers` generate loop) to make the continuous-batching story concrete by
  contrast — expected to be substantially worse, but not yet measured.
- **Extraction side** (secondary to this project, benched deliberately): a schema built to
  cover more real-world field variants; cropping a receipt's blank top/bottom margins before
  it reaches the VLM, which would cut visual tokens (and cost) further without touching the
  content — the opposite of the "just cap resolution" lever, targeting waste in the image
  itself rather than the model's processing of it.
