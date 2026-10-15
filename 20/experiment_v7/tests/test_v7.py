from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core_v7 import write_synthetic_checks
from extract_physical_features_v7 import assert_truth_isolation as extract_guard
from apply_pseudolabels_v7 import assert_truth_isolation as apply_guard


def test_truth_isolation_guards():
    extract_guard(); apply_guard()


def test_synthetic_mechanisms(tmp_path):
    result = write_synthetic_checks(tmp_path / "checks.json")
    assert result["source_delayed_sidelobe"]["pass"]
    assert result["independent_reflection"]["pass"]
    assert result["strict_equivalence"]["pass"]
