"""
Scoring functions for evaluation and preference data generation.
Includes automatic metrics and clinical-style rubrics.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import numpy as np


# ============================================================================
# AUTOMATIC TEXT METRICS
# ============================================================================

def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
    """
    Compute BLEU scores (simplified implementation).
    For production, consider using sacrebleu or nltk.
    """
    from collections import defaultdict
    
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    scores = {}
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        hyp_ngrams = get_ngrams(hypothesis, n)
        
        matches = sum((hyp_ngrams & ref_ngrams).values())
        total = max(sum(hyp_ngrams.values()), 1)
        
        scores[f"bleu_{n}"] = matches / total if total > 0 else 0.0
    
    # Approximate corpus BLEU (geometric mean)
    if all(scores[f"bleu_{n}"] > 0 for n in range(1, max_n + 1)):
        scores["bleu"] = bp * np.exp(np.mean([np.log(scores[f"bleu_{n}"]) for n in range(1, max_n + 1)]))
    else:
        scores["bleu"] = 0.0
    
    return scores


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE scores (simplified implementation).
    For production, consider using rouge-score package.
    """
    def get_ngrams(text: str, n: int) -> set:
        tokens = text.lower().split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def rouge_n(ref_ngrams: set, hyp_ngrams: set) -> float:
        if not ref_ngrams:
            return 0.0
        overlap = len(ref_ngrams & hyp_ngrams)
        return overlap / len(ref_ngrams)
    
    def rouge_l(reference: str, hypothesis: str) -> float:
        """ROUGE-L (Longest Common Subsequence)."""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # LCS length
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs = dp[m][n]
        if m == 0:
            return 0.0
        return lcs / m
    
    scores = {}
    
    # ROUGE-1
    ref_1grams = get_ngrams(reference, 1)
    hyp_1grams = get_ngrams(hypothesis, 1)
    scores["rouge_1"] = rouge_n(ref_1grams, hyp_1grams)
    
    # ROUGE-2
    ref_2grams = get_ngrams(reference, 2)
    hyp_2grams = get_ngrams(hypothesis, 2)
    scores["rouge_2"] = rouge_n(ref_2grams, hyp_2grams)
    
    # ROUGE-L
    scores["rouge_l"] = rouge_l(reference, hypothesis)
    
    return scores


def compute_structural_metrics(text: str) -> Dict[str, Any]:
    """
    Compute structural metrics for radiology reports.
    
    Checks:
    - Presence of "Impression:" section
    - Presence of "Vascular summary:" or similar
    - Length constraints
    - Section ordering
    """
    metrics = {}
    
    # Check for Impression section
    has_impression = bool(re.search(r'impression\s*:', text, re.IGNORECASE))
    metrics["has_impression_section"] = has_impression
    
    # Check for Vascular summary section
    has_vascular = bool(re.search(r'vascular\s+(summary|findings)\s*:', text, re.IGNORECASE))
    metrics["has_vascular_section"] = has_vascular
    
    # Length metrics
    word_count = len(text.split())
    char_count = len(text)
    metrics["word_count"] = word_count
    metrics["char_count"] = char_count
    metrics["within_length_limits"] = 20 <= word_count <= 500
    
    # Section ordering (Vascular before Impression)
    vascular_pos = re.search(r'vascular\s+(summary|findings)\s*:', text, re.IGNORECASE)
    impression_pos = re.search(r'impression\s*:', text, re.IGNORECASE)
    
    if vascular_pos and impression_pos:
        metrics["correct_section_order"] = vascular_pos.start() < impression_pos.start()
    else:
        metrics["correct_section_order"] = None
    
    # Overall structure score
    structure_score = 0.0
    if has_impression:
        structure_score += 0.3
    if has_vascular:
        structure_score += 0.3
    if metrics["within_length_limits"]:
        structure_score += 0.2
    if metrics["correct_section_order"]:
        structure_score += 0.2
    
    metrics["structure_score"] = structure_score
    
    return metrics


# ============================================================================
# CLINICAL RUBRICS
# ============================================================================

class ClinicalRubricScorer:
    """
    Clinical-style rubric scorer for stroke radiology reports.
    
    Implements:
    - Hallucination detection
    - Consistency checks (left/right)
    - Uncertainty language assessment
    """
    
    # Anatomical regions for stroke
    VESSEL_NAMES = [
        "aca", "anterior cerebral artery",
        "mca", "middle cerebral artery",
        "pca", "posterior cerebral artery",
        "ica", "internal carotid",
        "vertebral", "basilar",
        "posterior circulation"
    ]
    
    LATERALITY_TERMS = ["left", "right", "bilateral", "midline"]
    
    OVERCONFIDENT_WORDS = [
        "definitely", "certainly", "absolutely", "clearly",
        "unambiguous", "without doubt", "undeniably"
    ]
    
    UNCERTAINTY_WORDS = [
        "possible", "possibly", "may", "might", "could",
        "suggestive", "suspicious", "likely", "probably",
        "cannot exclude", "not excluded"
    ]
    
    def __init__(self, use_llm_judge: bool = False, llm_judge_model: Optional[str] = None):
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model
    
    def score(self, prediction: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute all clinical rubric scores.
        
        Args:
            prediction: Model-generated text
            reference: Ground truth text (optional, for comparison)
        
        Returns:
            Dict with all rubric scores
        """
        scores = {
            "hallucination": self.check_hallucination(prediction, reference),
            "consistency": self.check_consistency(prediction),
            "uncertainty": self.check_uncertainty(prediction),
        }
        
        # Overall clinical score (weighted average)
        weights = {"hallucination": 0.4, "consistency": 0.35, "uncertainty": 0.25}
        overall = sum(scores[k]["score"] * weights[k] for k in weights)
        scores["overall_clinical_score"] = overall
        
        return scores
    
    def check_hallucination(
        self,
        prediction: str,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check for hallucinated vessel/site claims.
        
        Returns:
            Dict with hallucination flags and score
        """
        result = {
            "flags": [],
            "score": 1.0,  # Higher is better (no hallucination)
            "details": {}
        }
        
        pred_lower = prediction.lower()
        
        # Extract mentioned vessels and sites
        mentioned_vessels = []
        for vessel in self.VESSEL_NAMES:
            if vessel in pred_lower:
                mentioned_vessels.append(vessel)
        
        result["details"]["mentioned_vessels"] = mentioned_vessels
        
        # If reference provided, check for unsupported claims
        if reference:
            ref_lower = reference.lower()
            
            # Check for vessels mentioned in pred but not in ref
            ref_vessels = [v for v in self.VESSEL_NAMES if v in ref_lower]
            unsupported = [v for v in mentioned_vessels if v not in ref_vessels]
            
            if unsupported:
                result["flags"].append(f"Potentially unsupported vessels: {unsupported}")
                result["score"] -= 0.2 * len(unsupported)
        
        # Check for contradictory statements
        contradictions = self._detect_contradictions(prediction)
        if contradictions:
            result["flags"].extend(contradictions)
            result["score"] -= 0.3 * len(contradictions)
        
        result["score"] = max(0.0, min(1.0, result["score"]))
        return result
    
    def check_consistency(self, prediction: str) -> Dict[str, Any]:
        """
        Check for consistency issues (left/right mismatches, contradictions).
        
        Returns:
            Dict with consistency flags and score
        """
        result = {
            "flags": [],
            "score": 1.0,
            "details": {}
        }
        
        pred_lower = prediction.lower()
        
        # Check for left/right consistency
        left_count = pred_lower.count("left")
        right_count = pred_lower.count("right")
        
        result["details"]["left_mentions"] = left_count
        result["details"]["right_mentions"] = right_count
        
        # Check for same vessel mentioned with different lateralities
        lines = prediction.split('\n')
        vessel_laterality = {}
        
        for line in lines:
            line_lower = line.lower()
            for vessel in self.VESSEL_NAMES:
                if vessel in line_lower:
                    # Extract laterality for this vessel
                    lat = None
                    if "left " + vessel in line_lower or vessel + " left" in line_lower:
                        lat = "left"
                    elif "right " + vessel in line_lower or vessel + " right" in line_lower:
                        lat = "right"
                    
                    if vessel in vessel_laterality and lat != vessel_laterality[vessel]:
                        if lat and vessel_laterality[vessel]:
                            result["flags"].append(
                                f"Laterality mismatch for {vessel}: "
                                f"{vessel_laterality[vessel]} vs {lat}"
                            )
                            result["score"] -= 0.25
                    
                    if lat:
                        vessel_laterality[vessel] = lat
        
        # Check for impossible combinations
        if "bilateral" in pred_lower:
            # Should not have specific left/right after bilateral
            bilateral_pos = pred_lower.find("bilateral")
            after_bilateral = pred_lower[bilateral_pos:bilateral_pos + 200]
            if "left" in after_bilateral or "right" in after_bilateral:
                result["flags"].append("Specific laterality mentioned after 'bilateral'")
                result["score"] -= 0.15
        
        result["score"] = max(0.0, min(1.0, result["score"]))
        return result
    
    def check_uncertainty(self, prediction: str) -> Dict[str, Any]:
        """
        Check for appropriate uncertainty language.
        
        Returns:
            Dict with uncertainty flags and score
        """
        result = {
            "flags": [],
            "score": 1.0,
            "details": {}
        }
        
        pred_lower = prediction.lower()
        
        # Count overconfident language
        overconfident_count = sum(1 for word in self.OVERCONFIDENT_WORDS if word in pred_lower)
        result["details"]["overconfident_words"] = overconfident_count
        
        if overconfident_count > 0:
            result["flags"].append(f"Overconfident language detected ({overconfident_count} instances)")
            result["score"] -= 0.1 * overconfident_count
        
        # Count uncertainty words (these are generally good in radiology)
        uncertainty_count = sum(1 for word in self.UNCERTAINTY_WORDS if word in pred_lower)
        result["details"]["uncertainty_words"] = uncertainty_count
        
        # Check for appropriate uncertainty in impression
        impression_match = re.search(r'impression\s*:(.+?)(?=\n\w+:|$)', prediction, re.IGNORECASE | re.DOTALL)
        if impression_match:
            impression = impression_match.group(1).lower()
            has_uncertainty = any(word in impression for word in self.UNCERTAINTY_WORDS)
            result["details"]["impression_has_uncertainty"] = has_uncertainty
            
            # For stroke reports, some uncertainty is appropriate
            if not has_uncertainty and "occlusion" in impression:
                # Strong claim without uncertainty - flag for review
                result["flags"].append("Definitive occlusion claim without uncertainty qualifier")
        
        result["score"] = max(0.0, min(1.0, result["score"]))
        return result
    
    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect contradictory statements in text."""
        contradictions = []
        text_lower = text.lower()
        
        # Check for "no abnormality" + specific findings
        no_abnormality = any(phrase in text_lower for phrase in [
            "no abnormality", "normal", "unremarkable"
        ])
        
        has_findings = any(phrase in text_lower for phrase in [
            "occlusion", "stenosis", "thrombus", "embolus",
            "cutoff", "filling defect"
        ])
        
        if no_abnormality and has_findings:
            contradictions.append("Contradiction: 'no abnormality' with specific findings")
        
        # Check for acute + chronic (should specify which)
        if "acute" in text_lower and "chronic" in text_lower:
            if "acute on chronic" not in text_lower:
                contradictions.append("Both 'acute' and 'chronic' without clarification")
        
        return contradictions


# ============================================================================
# SCORING FUNCTIONS FOR PREFERENCE DATA
# ============================================================================

def rule_based_score(
    prediction: str,
    reference: Optional[str] = None
) -> float:
    """
    Rule-based scoring function for preference data generation.
    
    Combines structural and clinical rubric scores.
    """
    # Structural score
    structural = compute_structural_metrics(prediction)
    structural_score = structural["structure_score"]
    
    # Clinical rubric score
    rubric = ClinicalRubricScorer()
    clinical = rubric.score(prediction, reference)
    clinical_score = clinical["overall_clinical_score"]
    
    # Text similarity if reference available
    similarity_score = 0.5
    if reference:
        rouge = compute_rouge(reference, prediction)
        similarity_score = rouge["rouge_l"]
    
    # Weighted combination
    final_score = (
        0.3 * structural_score +
        0.4 * clinical_score +
        0.3 * similarity_score
    )
    
    return final_score


def llm_judge_score(
    prediction: str,
    prompt: str,
    images: Optional[List] = None,
    judge_model: Optional[str] = None
) -> float:
    """
    DEPRECATED text-only placeholder. Kept for backwards compatibility.

    For a real, image-grounded judge use `src.reward.LLMJudge`, which sends the
    CTA images to a multimodal model. This function intentionally raises if called
    without a backend so it can no longer silently inject random rewards.
    """
    raise NotImplementedError(
        "llm_judge_score is a non-functional placeholder. Use src.reward.LLMJudge "
        "(image-grounded) or src.reward.CompositeReward instead."
    )


# ============================================================================
# FACT-BASED (LABEL-GROUNDED) SCORING
# ============================================================================
#
# These functions score a generated report against the *structured case labels*
# (e.g. {"anomaly_present": true, "main_region": "Right MCA"}) rather than against
# the reference string. This makes the reward depend on diagnostic correctness
# (presence of large-vessel occlusion, correct vessel, correct side, calibrated
# uncertainty) instead of lexical overlap.

# Canonical vessel vocabulary. Each canonical key maps to surface forms / regexes.
# Order matters: more specific names are matched first.
_VESSEL_PATTERNS: List[Tuple[str, str]] = [
    ("ica", r"\b(ica|internal carotid(?: artery)?)\b"),
    ("mca", r"\b(mca|middle cerebral(?: artery)?|m1|m2)\b"),
    ("aca", r"\b(aca|anterior cerebral(?: artery)?|a1|a2)\b"),
    ("pca", r"\b(pca|posterior cerebral(?: artery)?|p1|p2)\b"),
    ("basilar", r"\b(basilar(?: artery)?)\b"),
    ("vertebral", r"\b(vertebral(?: artery)?|v4)\b"),
]

# Positive findings indicating a large-vessel / acute abnormality.
_POSITIVE_FINDING_PATTERNS = [
    r"\bocclusion\b", r"\boccluded\b", r"\bocclusive\b",
    r"\bthrombus\b", r"\bthrombosis\b", r"\bembol", r"\bclot\b",
    r"\bcut[- ]?off\b", r"\bfilling defect\b", r"\bnon[- ]?opacif",
    r"\blarge[- ]vessel occlusion\b", r"\blvo\b",
    r"\bhigh[- ]grade stenosis\b", r"\bflow[- ]limiting stenosis\b",
]

# Explicit negation / normal statements.
_NEGATIVE_FINDING_PATTERNS = [
    r"\bno (?:definite |evidence of )?(?:proximal |large[- ]vessel )?(?:occlusion|cutoff|filling defect|thrombus)\b",
    r"\bno large[- ]vessel occlusion\b",
    r"\bno lvo\b",
    r"\bpatent\b", r"\bunremarkable\b", r"\bno acute\b",
    r"\bno significant\b", r"\bwithin normal limits\b",
    r"\bnormal (?:vascular|opacification|caliber)\b",
]

_HEDGE_PATTERNS = [
    r"\blimited\b", r"\bpartial(?:ly)?\b", r"\bcorrelate\b",
    r"\bmay\b", r"\bmight\b", r"\bpossible\b", r"\bpossibly\b",
    r"\bsuggestive\b", r"\bcannot (?:exclude|rule out)\b",
    r"\bnot excluded\b", r"\bif clinically\b", r"\bequivocal\b",
]


def _find_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _detect_vessels(text: str) -> List[str]:
    found = []
    for key, pat in _VESSEL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE) and key not in found:
            found.append(key)
    return found


def _detect_side(text: str) -> Optional[str]:
    """Return 'left', 'right', 'bilateral', or None using word-boundary matches.

    `\\bright\\b` already excludes 'rightward'/'upright' (no word boundary inside
    the longer token), so no special-casing is needed.
    """
    t = text.lower()
    if re.search(r"\bbilateral\b", t):
        return "bilateral"
    has_left = bool(re.search(r"\bleft\b", t))
    has_right = bool(re.search(r"\bright\b", t))
    if has_left and has_right:
        return None  # ambiguous; caller decides
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


_NEGATION_CUE = re.compile(
    r"\b(no|not|without|negative|absence|absent|patent|unremarkable|normal|none|"
    r"rule out|ruled out|exclude[ds]?)\b",
    re.IGNORECASE,
)


def _polarity(text: str) -> Tuple[bool, bool]:
    """
    Clause-level polarity for positive findings.

    Splits the report into clauses and, for each clause containing a positive
    finding term, checks whether the *same clause* carries a negation cue. This
    correctly reads "No large vessel occlusion" as negative while reading
    "Acute occlusion of the right MCA" as affirmative.

    Returns (affirmative, negated_present).
    """
    affirmative = False
    negated_present = False
    for clause in re.split(r"[.;:,\n]", text):
        if not _find_any(_POSITIVE_FINDING_PATTERNS, clause):
            continue
        if _NEGATION_CUE.search(clause):
            negated_present = True
        else:
            affirmative = True
    return affirmative, negated_present


def parse_report_claims(text: str) -> Dict[str, Any]:
    """
    Extract structured claims from a free-text report.

    Returns dict with:
        asserts_positive: bool   (claims an occlusion/LVO-type finding)
        asserts_negative: bool   (explicitly states normal/no occlusion)
        vessels: List[str]       (canonical vessel keys mentioned)
        side: Optional[str]      ('left'|'right'|'bilateral'|None)
        hedged: bool             (uses calibrated uncertainty language)
        overconfident: int       (count of overconfident words)
    """
    affirmative, negated_present = _polarity(text)
    asserts_positive = affirmative
    asserts_negative = negated_present or _find_any(_NEGATIVE_FINDING_PATTERNS, text)

    return {
        "asserts_positive": asserts_positive,
        "asserts_negative": asserts_negative,
        "vessels": _detect_vessels(text),
        "side": _detect_side(text),
        "hedged": _find_any(_HEDGE_PATTERNS, text),
        "overconfident": sum(
            1 for w in ClinicalRubricScorer.OVERCONFIDENT_WORDS if w in text.lower()
        ),
    }


def parse_labels(labels: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize structured case labels into {anomaly, vessel, side}.

    Accepts flexible schemas, e.g.:
        {"anomaly_present": true, "main_region": "Right MCA"}
        {"anomaly_present": false}
        {"vessel": "mca", "side": "right", "anomaly_present": true}
    Returns {anomaly: Optional[bool], vessel: Optional[str], side: Optional[str]}.
    """
    out = {"anomaly": None, "vessel": None, "side": None}
    if not labels or not isinstance(labels, dict):
        return out

    if "anomaly_present" in labels:
        out["anomaly"] = bool(labels["anomaly_present"])
    elif "anomaly" in labels:
        out["anomaly"] = bool(labels["anomaly"])

    region_text = " ".join(
        str(labels.get(k, "")) for k in ("main_region", "region", "vessel", "side", "laterality")
    )
    vessels = _detect_vessels(region_text)
    if vessels:
        out["vessel"] = vessels[0]
    elif labels.get("vessel"):
        out["vessel"] = str(labels["vessel"]).lower()

    side = _detect_side(region_text)
    if side:
        out["side"] = side
    elif labels.get("side"):
        out["side"] = str(labels["side"]).lower()
    elif labels.get("laterality"):
        out["side"] = str(labels["laterality"]).lower()

    return out


def fact_based_report(
    prediction: str,
    labels: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Score a prediction against structured case labels (diagnostic correctness).

    Returns a dict:
        {
          "available": bool,        # False if labels carry no usable ground truth
          "score": float in [0,1],  # composite fact score (only meaningful if available)
          "components": {...},      # per-axis sub-scores
          "claims": {...},          # parsed prediction claims
          "labels": {...},          # normalized labels
        }

    Axes (weighted mean over the ones that are evaluable):
      - detection  (0.45): does presence/absence of LVO match the label?
      - vessel     (0.20): correct vessel (only when anomaly present & vessel labeled)
      - side       (0.25): correct laterality (only when anomaly present & side labeled)
      - calibration(0.10): penalize overconfidence; reward appropriate hedging
    (Laterality is weighted heavily because wrong-side errors mislead treatment.)
    A multiplicative safety factor penalizes internal contradictions.
    """
    gt = parse_labels(labels)
    claims = parse_report_claims(prediction)

    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    # --- Detection (presence/absence of large-vessel occlusion) ---
    if gt["anomaly"] is not None:
        weights["detection"] = 0.45
        if gt["anomaly"] is True:
            if claims["asserts_positive"]:
                components["detection"] = 1.0
            elif claims["asserts_negative"] and not claims["hedged"]:
                components["detection"] = 0.0   # confident miss (worst case)
            elif claims["hedged"]:
                components["detection"] = 0.5   # hedged, didn't commit
            else:
                components["detection"] = 0.3
        else:  # anomaly absent (normal)
            if claims["asserts_positive"]:
                components["detection"] = 0.0   # false positive
            elif claims["asserts_negative"]:
                components["detection"] = 1.0
            elif claims["hedged"]:
                components["detection"] = 0.7   # cautious, acceptable
            else:
                components["detection"] = 0.5

    # --- Vessel correctness (only meaningful for true positives) ---
    if gt["anomaly"] is True and gt["vessel"]:
        weights["vessel"] = 0.20
        if not claims["vessels"]:
            components["vessel"] = 0.3          # under-specified, not wrong
        elif gt["vessel"] in claims["vessels"]:
            components["vessel"] = 1.0
        else:
            components["vessel"] = 0.0          # named the wrong vessel

    # --- Side correctness (laterality errors are clinically serious) ---
    if gt["anomaly"] is True and gt["side"] in {"left", "right", "bilateral"}:
        weights["side"] = 0.25
        if claims["side"] is None:
            components["side"] = 0.3
        elif claims["side"] == gt["side"]:
            components["side"] = 1.0
        else:
            components["side"] = 0.0

    # --- Calibration (always evaluable) ---
    weights["calibration"] = 0.10
    cal = 1.0 - 0.25 * claims["overconfident"]
    if gt["anomaly"] is False and claims["hedged"]:
        cal = min(1.0, cal + 0.1)
    components["calibration"] = max(0.0, min(1.0, cal))

    available = any(k in weights for k in ("detection", "vessel", "side"))

    if not weights:
        return {"available": False, "score": 0.0, "components": {},
                "claims": claims, "labels": gt}

    total_w = sum(weights.values())
    score = sum(components[k] * weights[k] for k in weights) / total_w

    # Multiplicative safety penalty for internal contradictions.
    contradictions = ClinicalRubricScorer()._detect_contradictions(prediction)
    if contradictions:
        score *= max(0.0, 1.0 - 0.3 * len(contradictions))

    return {
        "available": available,
        "score": float(max(0.0, min(1.0, score))),
        "components": components,
        "weights": weights,
        "claims": claims,
        "labels": gt,
        "contradictions": contradictions,
    }


def fact_based_score(
    prediction: str,
    labels: Optional[Dict[str, Any]] = None,
    reference: Optional[str] = None,
) -> float:
    """
    Thin float wrapper around `fact_based_report`.

    If labels carry no usable ground truth, falls back to `rule_based_score`
    (structure + clinical rubric + ROUGE) so the pipeline still produces a value.
    """
    report = fact_based_report(prediction, labels)
    if report["available"]:
        return report["score"]
    return rule_based_score(prediction, reference)


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_scores(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scores across multiple examples."""
    if not scores:
        return {}
    
    aggregated = {}
    
    # Collect all numeric keys
    numeric_keys = set()
    for score in scores:
        for key, value in score.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)
    
    # Compute statistics
    for key in numeric_keys:
        values = [s[key] for s in scores if key in s]
        if values:
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
    
    return aggregated
