"""
Image-grounded reward components for preference-data generation.

This module contains the parts of the reward that make alignment depend on
*diagnostic correctness* rather than lexical overlap with the reference:

    - LLMJudge:        a multimodal LLM-as-judge that actually sees the CTA-MIP
                       images and scores correctness / grounding / laterality /
                       calibration. Pluggable backend; gracefully reports
                       "unavailable" when no backend/API key is configured.

    - CompositeReward: combines the fact-based (label-grounded) score, the LLM
                       judge, structural conformity, and (optionally) reference
                       similarity into a single scalar, renormalizing weights
                       over whichever components are available for a given case.

Design goals:
    * No silent random rewards. A component that cannot run returns
      available=False and is dropped from the weighted average.
    * No hard third-party dependency. The OpenAI-compatible backend is imported
      lazily; without it (or without an API key) the judge simply abstains.
    * Deterministic, cacheable judge calls (temperature 0, response cached per
      (case_id, text-hash)).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scoring import (
    compute_rouge,
    compute_structural_metrics,
    fact_based_report,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Image-grounded LLM judge
# ============================================================================

JUDGE_SYSTEM_PROMPT = (
    "You are a board-certified neuroradiologist grading an AI-generated report "
    "for a CTA-MIP (CT angiography maximum-intensity-projection) study of the head. "
    "You are shown the same slab images the AI saw. Grade ONLY what the images "
    "support. Reward correct detection of large-vessel occlusion, correct vessel "
    "and side, claims that are grounded in the images, and appropriately calibrated "
    "uncertainty. Penalize hallucinated findings, wrong laterality, overconfident "
    "language, and internal contradictions."
)

JUDGE_RUBRIC_INSTRUCTIONS = (
    "Return STRICT JSON with this schema and nothing else:\n"
    "{\n"
    '  "correctness": <0-1>,    // findings match what the images support\n'
    '  "grounding": <0-1>,      // every claim is supported by the images\n'
    '  "laterality": <0-1>,     // left/right/vessel correctness\n'
    '  "calibration": <0-1>,    // uncertainty is appropriate, not over/under-confident\n'
    '  "overall": <0-1>,        // holistic clinical quality\n'
    '  "rationale": "<one sentence>"\n'
    "}\n"
)


@dataclass
class JudgeResult:
    available: bool
    score: float = 0.0
    subscores: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    error: Optional[str] = None


class LLMJudge:
    """
    Multimodal LLM-as-judge.

    Backend: any OpenAI-compatible chat-completions endpoint that accepts image
    content parts (e.g. OpenAI GPT-4o, or a local vLLM/Ollama server exposing the
    OpenAI API). Configure via constructor args or environment variables:

        LLM_JUDGE_MODEL      (e.g. "gpt-4o-2024-11-20")
        LLM_JUDGE_BASE_URL   (optional, for self-hosted / Azure / proxies)
        OPENAI_API_KEY       (or pass api_key=...)

    If the `openai` package is missing OR no model/key is configured, the judge
    abstains (every call returns JudgeResult(available=False)) so the pipeline
    can fall back to the fact-based score.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_images: int = 4,
        weight_overall: float = 0.5,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        self.model = model or os.environ.get("LLM_JUDGE_MODEL")
        self.base_url = base_url or os.environ.get("LLM_JUDGE_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_images = max_images
        self.weight_overall = weight_overall
        self.temperature = temperature
        self.timeout = timeout
        self._cache: Dict[str, JudgeResult] = {}
        self._client = None
        self._init_error: Optional[str] = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.model or not self.api_key:
            self._init_error = "LLM judge not configured (missing model or API key)."
            logger.warning("%s Judge will abstain.", self._init_error)
            return
        try:
            from openai import OpenAI  # lazy import
        except Exception as exc:  # pragma: no cover - env dependent
            self._init_error = f"openai package unavailable: {exc}"
            logger.warning("%s Judge will abstain.", self._init_error)
            return
        kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    @property
    def is_available(self) -> bool:
        return self._client is not None

    # --- image encoding -----------------------------------------------------
    @staticmethod
    def _encode_image(path: Path) -> Optional[str]:
        try:
            data = path.read_bytes()
        except Exception as exc:
            logger.warning("Could not read image %s: %s", path, exc)
            return None
        ext = path.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/{mime};base64,{b64}"

    def _build_messages(
        self, prompt: str, prediction: str, image_data_urls: List[str]
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"CASE PROMPT:\n{prompt}\n\n"
                    f"AI-GENERATED REPORT TO GRADE:\n{prediction}\n\n"
                    f"{JUDGE_RUBRIC_INSTRUCTIONS}"
                ),
            }
        ]
        for url in image_data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _cache_key(case_id: str, prediction: str) -> str:
        h = hashlib.sha256(prediction.encode("utf-8")).hexdigest()[:16]
        return f"{case_id}:{h}"

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        # Tolerate code fences / extra prose around the JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object in judge response: {text[:200]!r}")
        return json.loads(text[start : end + 1])

    def score(
        self,
        prediction: str,
        prompt: str,
        image_paths: List[str],
        images_root: str,
        case_id: str = "case",
    ) -> JudgeResult:
        if not self.is_available:
            return JudgeResult(available=False, error=self._init_error)

        key = self._cache_key(case_id, prediction)
        if key in self._cache:
            return self._cache[key]

        root = Path(images_root)
        data_urls: List[str] = []
        for p in image_paths[: self.max_images]:
            url = self._encode_image(root / p)
            if url:
                data_urls.append(url)
        if not data_urls:
            result = JudgeResult(available=False, error="no decodable images")
            self._cache[key] = result
            return result

        messages = self._build_messages(prompt, prediction, data_urls)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            raw = resp.choices[0].message.content or ""
            parsed = self._parse_response(raw)
        except Exception as exc:
            logger.warning("Judge call failed for %s: %s", case_id, exc)
            result = JudgeResult(available=False, error=str(exc))
            self._cache[key] = result
            return result

        def _clip(v: Any) -> float:
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return 0.0

        subscores = {
            k: _clip(parsed.get(k))
            for k in ("correctness", "grounding", "laterality", "calibration", "overall")
        }
        # Blend the holistic 'overall' with the mean of the specific axes.
        specific = [subscores[k] for k in ("correctness", "grounding", "laterality", "calibration")]
        mean_specific = sum(specific) / len(specific)
        final = self.weight_overall * subscores["overall"] + (1 - self.weight_overall) * mean_specific

        result = JudgeResult(
            available=True,
            score=float(max(0.0, min(1.0, final))),
            subscores=subscores,
            rationale=str(parsed.get("rationale", "")),
        )
        self._cache[key] = result
        return result


# ============================================================================
# Composite reward
# ============================================================================

@dataclass
class CompositeWeights:
    fact: float = 0.5
    judge: float = 0.3
    structure: float = 0.1
    similarity: float = 0.1


class CompositeReward:
    """
    Combine fact-based, judge, structural, and similarity signals.

    Weights are renormalized over the components that are actually available for
    a given candidate, so a case with no labels still produces a sensible score
    from judge + structure (+ similarity), and a run with no judge configured
    still works from fact + structure (+ similarity).

    Usage:
        reward = CompositeReward(judge=LLMJudge())          # judge may abstain
        s = reward.score(prediction, reference=ref, labels=lbl,
                         prompt=prompt, image_paths=imgs, images_root=root,
                         case_id=cid)
    """

    def __init__(
        self,
        weights: Optional[CompositeWeights] = None,
        judge: Optional[LLMJudge] = None,
        use_similarity: bool = True,
    ):
        self.weights = weights or CompositeWeights()
        self.judge = judge
        self.use_similarity = use_similarity

    def score_detailed(
        self,
        prediction: str,
        reference: Optional[str] = None,
        labels: Optional[Dict[str, Any]] = None,
        prompt: str = "",
        image_paths: Optional[List[str]] = None,
        images_root: Optional[str] = None,
        case_id: str = "case",
    ) -> Dict[str, Any]:
        comps: Dict[str, float] = {}
        w: Dict[str, float] = {}
        detail: Dict[str, Any] = {}

        # Fact-based (label-grounded)
        fact = fact_based_report(prediction, labels)
        detail["fact"] = fact
        if fact["available"]:
            comps["fact"] = fact["score"]
            w["fact"] = self.weights.fact

        # Image-grounded judge
        if self.judge is not None and self.judge.is_available and image_paths and images_root:
            jr = self.judge.score(prediction, prompt, image_paths, images_root, case_id)
            detail["judge"] = jr
            if jr.available:
                comps["judge"] = jr.score
                w["judge"] = self.weights.judge

        # Structural conformity
        structural = compute_structural_metrics(prediction)
        comps["structure"] = structural["structure_score"]
        w["structure"] = self.weights.structure
        detail["structure"] = structural

        # Reference similarity (weak anchor; kept small)
        if self.use_similarity and reference:
            sim = compute_rouge(reference, prediction)["rouge_l"]
            comps["similarity"] = sim
            w["similarity"] = self.weights.similarity
            detail["similarity"] = sim

        total_w = sum(w.values())
        if total_w == 0:
            final = 0.0
        else:
            final = sum(comps[k] * w[k] for k in w) / total_w

        return {
            "score": float(max(0.0, min(1.0, final))),
            "components": comps,
            "weights_used": w,
            "detail": detail,
        }

    def score(self, *args, **kwargs) -> float:
        return self.score_detailed(*args, **kwargs)["score"]
