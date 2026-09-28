# Phase 8 — the managed-baseline comparison (SROIE)

**Setup.** SROIE's 347-receipt test split, scored on the 4 fields Document Intelligence's
prebuilt-receipt model actually returns: `company`, `address`, `date`, `total`. Both systems
run single-shot (no repair loop, no retry-into-a-different-answer) — DI has no equivalent to
`pipeline.py`'s bounded repair, so giving the self-hosted side one here would make the
latency/cost comparison uneven by construction, not just inaccurate. Self-hosted config is
Qwen3-VL-4B-Instruct, reduced resolution (Phase 6's best-measured-F1 config on CORD, D-022) —
picked deliberately over 4B default so this comparison also doubles as "is the cheaper,
faster config still a fair fight against a managed service," not just "is self-hosting
possible at all." DI runs against Azure AI Document Intelligence's `prebuilt-receipt` model,
API version `2024-11-30`.

**A note on how this write-up came to look the way it does.** The first version of this table
scored `address` with the same character-level fuzzy match used for CORD's `name` field, and
found a huge, suspicious gap (self-hosted 0.974 vs. DI 0.498). Spot-checking 4 mismatches
showed DI's structured address object gets rejoined into a fixed field order that scrambles
word order relative to how the address is printed — real content, wrong shape. A systematic
check of all 164 DI address "mismatches" under the old metric confirmed this was real but only
*half* the story: 86 (52%) had high word overlap with gold (the reordering artifact), but 78
(48%) were genuinely low-overlap — real misses, not formatting noise. A word-overlap scorer
(`addresses_match`, Jaccard on lowercased alphanumeric tokens, threshold 0.7 — the same
threshold validated in that check) replaced the character-level default for this one field.
**That fix changed the headline finding, not just the address row** — see below.

## The table

| System | Scored | Coverage gap | Overall F1 | company | address | date | total | p50 latency | p95 latency | $/1k docs |
|---|---|---|---|---|---|---|---|---|---|---|
| **Self-hosted (4B, reduced-res)** | 345/347 | 2 (bad gold) | **0.871** | 0.925 | 0.939 | 0.794 | 0.826 | 3893ms | 5474ms | **$0.91** |
| **Azure DI (prebuilt-receipt)** | 326/347 | 21 (DI misses) | 0.871 | 0.816 | 0.743 | **0.986** | **0.939** | 3773ms | 5481ms | $10.00 |

Overall F1 is a **dead heat** — this is the corrected number, not the original 0.880 vs. 0.810.
The address fix moved DI from 0.498→0.743 and self-hosted from 0.974→0.939 (a looser metric
costs the already-strong side a little too), which was enough to erase what had looked like a
clear self-hosted accuracy lead. Coverage gaps unchanged from before: the self-hosted side's 2
are excluded receipts with a negative gold `total` (a real SROIE data-quality gap, same
category as CORD's own missing-total rows — D-017); DI's 21 are receipts where it genuinely
returned no `Total` field at all (confirmed from the raw response — some had a `Subtotal` but
no top-level `Total`), a real extraction miss, not a data problem. $/1k docs: self-hosted is
this run's own measured wall-clock cost (single-shot, sequential, 4B/reduced-res, $0.000222/
GPU-second — PROJECT_BRIEF.md §3); DI is Azure's published S0 rate, not what this specific run
actually paid (the real run mixed a free F0 tier with a paid S0 tier for reasons specific to
running an eval quickly, not something a real deployment would hit at steady state).

## Findings

**Overall accuracy is a genuine tie — the real differentiators are cost and coverage, not
accuracy.** This is the headline correction of this phase. Before fixing the address metric,
it looked like self-hosted won on every axis; it doesn't. What actually separates the two
systems is what they cost and whether they answer at all, not how often they're right.

**Self-hosted never fails to answer; DI does, 6% of the time.** Every one of the 345 scoreable
self-hosted predictions produced *some* value for every field — sometimes wrong, never absent.
DI's structured extraction pipeline simply has no `Total` field on 21/347 receipts, with no
repair path for that failure mode. A VLM's pattern-completion behavior, under schema-
constrained decoding, structurally cannot decline to fill a required field the way a
discriminative per-field detector can just not fire. This remains the most robust finding in
this run — it didn't move when the address metric was fixed, because it isn't a scoring
artifact, it's a count of a literally-missing field in DI's own raw response.

**DI clearly wins on `date` and `total`** — the two fields its structured, typed extraction is
actually built for (`TransactionDate` as a real date type, `Total` as a currency-typed
amount), and this held before and after the address fix. This is DI's home turf inside its
home turf: English receipts, and the two fields a receipt-specific model is tuned hardest on.

**`address` is now a real, if narrower, self-hosted edge (0.939 vs. 0.743)**, not the
misleading 2x gap the character-level metric first suggested. Of DI's original 164 "misses,"
roughly half were content-correct-but-reordered and are now scored correctly; the other half
were genuine misses (one receipt where DI extracted no address at all, several where DI
returned a different, wrong address entirely) and are still scored as misses under the new
metric too. The corrected number is closer to the truth, not just kinder to DI.

**Latency is close to a wash** (3893ms vs 3773ms p50) — neither system has a real edge here at
single-request granularity; the story is entirely in cost and coverage, not speed or, as it
turns out, overall accuracy either.

## Does the ranking flip between CORD (Indonesian) and SROIE (English)?

This is the question D-007 named as the reason to run SROIE at all — and it can only be
answered qualitatively here, not with a second measured number, because DI was deliberately
never run on CORD (D-007: Microsoft's own guidance says DI's language support doesn't
guarantee it handles locale-specific conventions, and running it on CORD anyway would flatter
the self-hosted side for reasons unrelated to real capability — an unfair comparison, not a
missing one).

With the corrected numbers, this question matters *more*, not less: SROIE is DI's best
possible showing in this project (English, standard formatting, its home turf), and even there
it only ties on overall accuracy while carrying a real 6% coverage gap self-hosted doesn't.
CORD is everything DI's own documentation flags as outside its guaranteed competence:
Indonesian, non-English, and (per this project's own Phase 4 findings, D-017) inconsistent
number formatting that took real effort to parse correctly even with full access to the raw
ground truth. A tie on DI's best day is not evidence DI would also tie on a locale it wasn't
built for — if anything, the coverage gap (a structural property of DI's extraction pipeline,
not something the address-metric fix touched at all) is the more locale-independent risk to
expect DI to carry into CORD, or worse there.

## When would you choose each — anchored in the data-residency framing (§1)

For a company where customer documents legally cannot leave company infrastructure — the
GDPR-driven constraint this project is actually built around — the corrected numbers make the
case for self-hosting *stronger*, not weaker, even though the accuracy story is now a tie
rather than a clear win: a company forced into self-hosting isn't giving up accuracy at all
under DI's best possible conditions (English, its home turf), and it's still getting full
coverage and an order of magnitude lower cost. The constraint costs nothing on accuracy and
saves real money — a better outcome than the original (accuracy-favoring) numbers even
suggested.

For a company with no data-residency constraint at all, the choice now hinges on cost and
coverage tolerance more than on accuracy, since accuracy is a wash. If the workload is
dominated by `date`/`total` extraction — expense reconciliation, where a wrong or missing
total is the expensive failure mode — DI's near-99% total F1, structured typed output, and
$10/1k-docs price (cheap for a team that wants zero GPU infrastructure) are a real, specific
strength. If the workload needs to *never* silently come back empty, or needs `company`/
`address` more than `date`/`total`, the self-hosted config's full coverage and 11x lower cost
win outright — assuming the team is willing to own the serving stack this project spent eight
phases building.

## Caveats — what this measurement does not show

- **One sample per request, no repeated trials, on either system** — same caveat Phase 6
  carried forward (vLLM's default temperature 0.7, never explicitly set). F1 differences of a
  point or two are not distinguishable from sampling noise on this evidence alone — and with
  the corrected numbers now an exact tie to three decimal places (0.871 = 0.871), that caveat
  matters more than it did before the fix, not less.
- **DI's cost is the published S0 rate, not this run's actual (partly free) cost** — stated
  deliberately as the production-realistic number, not the number that would make this
  comparison look most favorable.
- **`no_total` was scored as a DI failure, but the underlying receipts weren't manually
  audited** to rule out (e.g.) a genuinely blank total on an unusual receipt layout. Plausible
  but not confirmed to be entirely DI's fault in every one of the 21 cases.
- **This is one config vs. one config** — 4B/reduced-res, not the full four-config matrix
  against DI. Phase 6's own table shows 4B/default-res scores slightly lower on CORD than
  4B/reduced-res; a second SROIE run at 4B/default-res (for a number directly comparable to
  Phase 6's CORD headline row) was deliberately deferred, not forgotten.
- **The address word-overlap threshold (0.7) was validated against DI's actual mismatch
  distribution, not chosen a priori** — a reasonable, evidence-based choice, but not something
  independently cross-validated against a held-out set; a different threshold could move the
  address numbers somewhat, though unlikely enough to erase the now-narrow gap entirely.

## What we'd try next, given more time

- Run the second self-hosted config (4B, default-res) on SROIE for a number directly
  comparable to Phase 6's CORD headline row, completing the 2x2 story across both datasets.
  Deliberately deferred this phase, not blocking.
- A small manual audit of DI's 21 `no_total` receipts, to confirm none are a genuinely blank
  or illegible total rather than a DI miss.
- A repeated-trial run (multiple samples per receipt, both systems) to put real error bars on
  an overall-F1 gap that is now, on point estimates alone, exactly zero.
