"""Tests for eval/cord.py's loader - the one piece of the Phase 4 accuracy harness that
doesn't depend on the normalization/matching/scoring TODOs. Reads the real committed
CORD test parquet (no GPU, no network - PROJECT_BRIEF.md §6: data/cord/ ships in-repo).
"""

from docintel.eval.cord import CordSample, load_cord_test_split


def test_loads_all_100_test_split_receipts() -> None:
    samples = list(load_cord_test_split())
    assert len(samples) == 100


def test_sample_shape() -> None:
    first = next(load_cord_test_split())
    assert isinstance(first, CordSample)
    assert first.image_id == 0
    assert isinstance(first.image_bytes, bytes)
    assert len(first.image_bytes) > 0
    # gt_parse always has at least a total block - checked across the full split in the
    # loader itself would be redundant; this just confirms the JSON navigation
    # (`json.loads(...)["gt_parse"]`) is unwrapping to the right nesting level.
    assert "total" in first.raw_ground_truth


def test_image_ids_are_unique_and_ordered() -> None:
    samples = list(load_cord_test_split())
    ids = [s.image_id for s in samples]
    assert ids == list(range(100))
