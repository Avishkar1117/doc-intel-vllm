"""Per-request cost estimator - the seed for Phase 6/8's real aggregate €/1k-docs model.

Attributes one request's own wall-clock latency to the Modal L4 hourly rate. This is a
documented approximation: Modal bills GPU wall-clock time for the whole container, shared
across however many requests are actually batched together at that moment, so a single
request's "cost" here overstates the true marginal cost under concurrency. Phase 6's
throughput harness divides total GPU wall-clock time by documents actually served
concurrently instead - this module is not that; it just gives Phase 3's DoD ("container
runs end-to-end") something honest to report per request before that instrument exists.
"""

# PROJECT_BRIEF.md §3: L4 on Modal's Starter plan, $0.000222/sec (~$0.80/hr). USD, not EUR -
# the repo has no documented USD->EUR rate anywhere, and inventing one here would be exactly
# the kind of unstamped, version-sensitive number CLAUDE.md rule 5 warns against. The headline
# €/1k-docs table (§2) gets a real FX figure, sourced and dated, at Phase 6/8 - flagged as an
# open item there, not decided here.
MODAL_L4_USD_PER_SEC = 0.000222


def estimate_request_cost_usd(
    latency_ms: float, gpu_rate_usd_per_sec: float = MODAL_L4_USD_PER_SEC
) -> float:
    """Estimated Modal GPU cost attributable to one request, from its own measured latency."""
    return (latency_ms / 1000.0) * gpu_rate_usd_per_sec
