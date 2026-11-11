"""
Cross-encoder reranker backed by BGE-reranker-v2-m3.

Only loaded when ``config.USE_RERANKER`` is True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from config import RERANKER_KEEP, RERANKER_MODEL_NAME, RERANKER_TOP_N

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single reranked item."""

    index: int       # position in the *original* candidate list
    score: float     # cross-encoder relevance score


class Reranker:
    """Thin wrapper around a CrossEncoder for reranking retrieved chunks.

    Parameters
    ----------
    model_name : str
        HuggingFace model id for the cross-encoder.
    top_n : int
        Number of candidates to consider from the initial retrieval.
    keep : int
        Number of results to return after reranking.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        top_n: int = RERANKER_TOP_N,
        keep: int = RERANKER_KEEP,
    ) -> None:
        logger.info("Loading reranker model: %s (CPU) …", model_name)
        self.model = CrossEncoder(model_name, device="cpu")
        self.top_n = top_n
        self.keep = keep
        logger.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_n: int | None = None,
        keep: int | None = None,
    ) -> list[RerankResult]:
        """Score *query* against the first ``top_n`` passages and return the
        ``keep`` best results sorted by descending score.

        Parameters
        ----------
        query : str
            The user question.
        passages : list[str]
            Candidate passage texts (only the first ``top_n`` are scored).
        top_n : int, optional
            Override instance ``top_n``.
        keep : int, optional
            Override instance ``keep``.

        Returns
        -------
        list[RerankResult]
            Sorted from highest to lowest score.
        """
        top_n = top_n or self.top_n
        keep = keep or self.keep

        candidates = passages[:top_n]
        pairs = [[query, p] for p in candidates]

        scores = self.model.predict(pairs, show_progress_bar=False)

        indexed = [
            RerankResult(index=i, score=float(s))
            for i, s in enumerate(scores)
        ]
        indexed.sort(key=lambda r: r.score, reverse=True)
        return indexed[:keep]


# ──────────────────────────────────────────────
# Singleton helpers
# ──────────────────────────────────────────────

_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    """Lazily load the reranker singleton (non-Streamlit contexts)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


def get_cached_reranker():
    """Streamlit-cached reranker loader."""
    import streamlit as st

    @st.cache_resource(show_spinner="Loading reranker model …")
    def _load() -> Reranker:
        return Reranker()

    return _load()
