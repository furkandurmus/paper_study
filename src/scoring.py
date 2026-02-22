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
        "pca", "posterior circulation"
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
    Placeholder for LLM-as-judge scoring.
    
    TODO: Implement actual LLM judge integration.
    """
    # Placeholder - returns random score for now
    # In production, this would call an LLM API
    import random
    return random.uniform(0.5, 1.0)


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
