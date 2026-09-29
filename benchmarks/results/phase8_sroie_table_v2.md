| system | n_total | n_scored_all_fields | n_total_field_only | n_no_output | n_gold_invalid | overall_precision | overall_recall | overall_f1 | latency_p50_ms | latency_p95_ms | usd_per_1k_docs | address_f1 | company_f1 | date_f1 | total_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self-hosted (Qwen3-VL-4B, reduced-res) | 347 | 345 | 0 | 0 | 2 | 0.914 | 0.914 | 0.914 | 3893 | 5474 | 0.91 | 0.939 | 0.925 | 0.965 | 0.826 |
| Azure DI (prebuilt-receipt, S0 rate) | 347 | 326 | 4 | 15 | 2 | 0.869 | 0.868 | 0.868 | 3773 | 5481 | 10.0 | 0.743 | 0.816 | 0.986 | 0.927 |
