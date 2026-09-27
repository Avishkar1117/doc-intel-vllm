"""Phase 6: list every CORD test receipt the sweep may send.

Phase 5 filtered to the 83 receipts that reached the model, because D-014's encoder-cache
ceiling rejected the other 17. The Phase 6 probe showed that ceiling disappears without the
`max_pixels` launch flag (100/100 accepted), so the workload is now the full split - the
17 heaviest receipts included.
"""

import json
from pathlib import Path

from docintel.eval.cord import load_cord_test_split

MANIFEST_PATH = Path("benchmarks/results/phase6_workload.json")


def main() -> None:
    ids = [sample.image_id for sample in load_cord_test_split()]
    manifest = {"source": "cord_test_split_all", "image_ids": ids}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"{len(ids)} receipts -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
