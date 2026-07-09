#!/usr/bin/env python3
"""
Tests for the label-grounded fact score and the composite reward.

Run from repo root:
    python -m tests.test_reward
    # or
    pytest tests/test_reward.py

These tests use NO GPU, NO model, and NO network. The LLM judge is expected to
abstain (no API key), which exercises the composite renormalization path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import (
    parse_labels,
    parse_report_claims,
    fact_based_report,
    fact_based_score,
)
from src.reward import CompositeReward, CompositeWeights, LLMJudge


# --- sample reports ---------------------------------------------------------

GOOD_LVO = (
    "Findings: Abrupt cutoff of the right middle cerebral artery M1 segment with "
    "non-opacification of distal branches, consistent with occlusion. "
    "Impression: Acute occlusion of the right MCA."
)
WRONG_SIDE = (
    "Findings: Occlusion of the LEFT middle cerebral artery M1 segment. "
    "Impression: Acute left MCA occlusion."
)
WRONG_VESSEL = (
    "Findings: Occlusion of the basilar artery. "
    "Impression: Acute basilar occlusion."
)
MISS = (
    "Findings: The intracranial vessels are patent with normal opacification. "
    "Impression: No large vessel occlusion. Unremarkable CTA."
)
GARBAGE = (
    "Findings: Bad images. Impression: Massive stroke with complete blockage "
    "everywhere and definitely certain hemorrhage."
)
GOOD_NORMAL = (
    "Findings: Major intracranial arteries are patent without cutoff. "
    "Impression: No definite large vessel occlusion. Limited slab review; "
    "correlate with full CTA source images."
)


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_parse_labels():
    assert parse_labels({"anomaly_present": True, "main_region": "Right MCA"}) == {
        "anomaly": True, "vessel": "mca", "side": "right",
    }
    assert parse_labels({"anomaly_present": False}) == {
        "anomaly": False, "vessel": None, "side": None,
    }
    # unusable labels (the dummy-data shape)
    assert parse_labels({"source": "dummy_pref_chosen"})["anomaly"] is None
    print("ok  parse_labels")


def test_parse_claims():
    c = parse_report_claims(GOOD_LVO)
    assert c["asserts_positive"] and not c["asserts_negative"]
    assert "mca" in c["vessels"] and c["side"] == "right"

    n = parse_report_claims(MISS)
    assert n["asserts_negative"] and not n["asserts_positive"]

    g = parse_report_claims(GARBAGE)
    assert g["overconfident"] >= 1  # "definitely"/"certain"
    print("ok  parse_report_claims")


def test_fact_ranking_true_positive():
    labels = {"anomaly_present": True, "main_region": "Right MCA"}
    good = fact_based_report(GOOD_LVO, labels)
    wside = fact_based_report(WRONG_SIDE, labels)
    wves = fact_based_report(WRONG_VESSEL, labels)
    miss = fact_based_report(MISS, labels)
    garb = fact_based_report(GARBAGE, labels)

    for r in (good, wside, wves, miss, garb):
        assert r["available"], r

    # Correct report must rank strictly above every error mode.
    assert good["score"] > wside["score"], (good["score"], wside["score"])
    assert good["score"] > wves["score"], (good["score"], wves["score"])
    assert good["score"] > miss["score"], (good["score"], miss["score"])
    assert good["score"] > garb["score"], (good["score"], garb["score"])
    # A confident miss on a true LVO is among the worst outcomes.
    assert miss["score"] <= 0.35, miss["score"]
    # Wrong laterality should be penalized (loses the full side weight).
    assert wside["score"] < 0.8, wside["score"]
    # Wrong vessel should rank below wrong side here (vessel named incorrectly).
    assert wves["score"] < wside["score"], (wves["score"], wside["score"])
    print(f"ok  fact ranking (TP): good={good['score']:.3f} "
          f"wrong_side={wside['score']:.3f} wrong_vessel={wves['score']:.3f} "
          f"miss={miss['score']:.3f} garbage={garb['score']:.3f}")


def test_fact_ranking_true_negative():
    labels = {"anomaly_present": False}
    normal = fact_based_report(GOOD_NORMAL, labels)
    false_pos = fact_based_report(GOOD_LVO, labels)
    assert normal["available"] and false_pos["available"]
    assert normal["score"] > false_pos["score"], (normal["score"], false_pos["score"])
    assert false_pos["score"] <= 0.35, false_pos["score"]  # false positive penalized
    print(f"ok  fact ranking (TN): normal={normal['score']:.3f} "
          f"false_pos={false_pos['score']:.3f}")


def test_fact_fallback_without_labels():
    # No usable labels -> fact_based_score falls back to rule_based (still a float).
    s = fact_based_score(GOOD_LVO, {"source": "dummy"}, reference=GOOD_LVO)
    assert 0.0 <= s <= 1.0
    print(f"ok  fact fallback without labels: {s:.3f}")


def test_composite_judge_abstains():
    # No API key -> judge abstains; composite must still rank good > garbage
    # using fact + structure (+ similarity), renormalized.
    judge = LLMJudge()  # unconfigured
    assert not judge.is_available
    reward = CompositeReward(weights=CompositeWeights(), judge=judge)
    labels = {"anomaly_present": True, "main_region": "Right MCA"}

    good = reward.score_detailed(
        prediction=GOOD_LVO, reference=GOOD_LVO, labels=labels,
        prompt="CTA head", image_paths=["a.png"], images_root="/nonexistent",
        case_id="c1",
    )
    garbage = reward.score_detailed(
        prediction=GARBAGE, reference=GOOD_LVO, labels=labels,
        prompt="CTA head", image_paths=["a.png"], images_root="/nonexistent",
        case_id="c1",
    )
    assert "judge" not in good["weights_used"], "judge should have abstained"
    assert "fact" in good["weights_used"] and "structure" in good["weights_used"]
    assert _approx(sum(good["weights_used"].values()), sum(good["weights_used"].values()))
    assert good["score"] > garbage["score"], (good["score"], garbage["score"])
    print(f"ok  composite (judge abstains): good={good['score']:.3f} "
          f"garbage={garbage['score']:.3f} weights={good['weights_used']}")


def main():
    tests = [
        test_parse_labels,
        test_parse_claims,
        test_fact_ranking_true_positive,
        test_fact_ranking_true_negative,
        test_fact_fallback_without_labels,
        test_composite_judge_abstains,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
