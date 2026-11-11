"""
Gemini LLM wrapper using the official ``google-genai`` SDK.

Responsibilities
----------------
* Initialise the Gemini client from ``GOOGLE_API_KEY``.
* Structured JSON output via ``response_mime_type`` + ``response_schema``.
* Exponential-backoff retry for rate limits and transient errors.
* Follow-up question rewriting (standalone query generation).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass, field as _field  # type: ignore[no-redef]
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return _field()

from config import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL_NAME,
    GEMINI_RETRY_BASE_DELAY,
    GEMINI_TEMPERATURE,
)
from generation.prompts import REWRITE_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Pydantic response schema
# ──────────────────────────────────────────────

class AnswerResponse(BaseModel):
    """Schema enforced on every Gemini QA response."""

    answer: str = Field(description="The answer text in the requested language.")
    cited_chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of chunk_ids used to compose the answer.",
    )


# ──────────────────────────────────────────────
# Client initialisation
# ──────────────────────────────────────────────

def _get_client() -> Any:
    """Create a Gemini client using the ``GOOGLE_API_KEY`` env var."""
    if genai is None:
        raise ImportError(
            "google-genai SDK is not installed. Install with: pip install google-genai"
        )
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it before running the app."
        )
    return genai.Client(api_key=api_key)


# ──────────────────────────────────────────────
# Retry helper
# ──────────────────────────────────────────────

def _call_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = GEMINI_MAX_RETRIES,
    base_delay: float = GEMINI_RETRY_BASE_DELAY,
) -> Any:
    """Call ``generate_content`` with exponential-backoff retry.

    Retries on rate-limit (429), server errors (5xx), and transient
    network failures.  Non-retryable errors are raised immediately.

    Returns
    -------
    The raw ``GenerateContentResponse``.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            # Check for blocked / empty response
            if response is None or not response.text:
                logger.warning(
                    "Gemini returned empty/blocked response (attempt %d/%d)",
                    attempt, max_retries,
                )
                if attempt < max_retries:
                    _backoff(attempt, base_delay)
                    continue
                raise GeminiEmptyResponseError(
                    "Gemini returned an empty or blocked response after "
                    f"{max_retries} attempts."
                )
            return response

        except GeminiEmptyResponseError:
            raise

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            is_retryable = any(
                kw in exc_str
                for kw in ("429", "rate", "quota", "500", "503", "unavailable",
                           "timeout", "deadline", "resource_exhausted")
            )
            if is_retryable and attempt < max_retries:
                logger.warning(
                    "Gemini error (attempt %d/%d): %s – retrying …",
                    attempt, max_retries, exc,
                )
                _backoff(attempt, base_delay)
                continue
            raise

    # Should not reach here, but just in case
    raise last_exc or RuntimeError("Gemini call failed after retries.")


def _backoff(attempt: int, base_delay: float) -> None:
    delay = base_delay * (2 ** (attempt - 1))
    logger.info("Backing off %.1fs before retry …", delay)
    time.sleep(delay)


# ──────────────────────────────────────────────
# QA generation
# ──────────────────────────────────────────────

def generate_answer(
    question: str,
    context_chunks: list[dict[str, Any]],
    response_language: str,
    model: str = GEMINI_MODEL_NAME,
) -> AnswerResponse:
    """Generate a grounded answer using Gemini.

    Parameters
    ----------
    question : str
        The user's question (possibly rewritten to be standalone).
    context_chunks : list[dict]
        Each dict must have ``chunk_id``, ``text``, ``start``, ``end``.
    response_language : str
        The language the answer should be written in.
        ``"Same as my question"`` becomes ``"the same language as the question"``.
    model : str
        Gemini model name.

    Returns
    -------
    AnswerResponse
        Parsed ``{answer, cited_chunk_ids}`` from Gemini.
    """
    if response_language.lower().startswith("same"):
        response_language = "the same language as the question"

    # Format context
    context_str = _format_context(context_chunks)

    # Build the prompt
    prompt = SYSTEM_PROMPT.format(
        response_language=response_language,
        context=context_str,
        question=question,
    )

    client = _get_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnswerResponse,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )

    response = _call_with_retry(client, model, prompt, config)

    # Parse the response
    parsed = response.parsed
    if parsed is not None and isinstance(parsed, AnswerResponse):
        return parsed

    # Fallback: parse from raw text
    try:
        raw = json.loads(response.text)
        return AnswerResponse(**raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("Failed to parse Gemini response as JSON: %s", exc)
        logger.debug("Raw response: %s", response.text)
        return AnswerResponse(
            answer=response.text or "Sorry, I could not generate an answer.",
            cited_chunk_ids=[],
        )


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Format context chunks for the prompt."""
    parts: list[str] = []
    for c in chunks:
        start = c.get("start", 0)
        end = c.get("end", 0)
        chunk_id = c.get("chunk_id", "unknown")
        text = c.get("text", "")
        parts.append(
            f"[chunk_id: {chunk_id} | {start:.1f}s – {end:.1f}s]\n{text}"
        )
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# Follow-up question rewriting
# ──────────────────────────────────────────────

def rewrite_followup(
    question: str,
    history: list[dict[str, str]],
    model: str = GEMINI_MODEL_NAME,
) -> str:
    """Rewrite a follow-up question into a standalone query.

    Parameters
    ----------
    question : str
        The follow-up question.
    history : list[dict]
        Recent chat turns, each ``{"role": "user"|"assistant", "content": str}``.
    model : str
        Gemini model name.

    Returns
    -------
    str
        The standalone question (in the same language as the input).
    """
    history_str = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
    )
    prompt = REWRITE_PROMPT.format(history=history_str, question=question)

    client = _get_client()
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=512,
    )

    try:
        response = _call_with_retry(client, model, prompt, config)
        rewritten = response.text.strip()
        if rewritten:
            logger.info("Rewritten question: %s → %s", question, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Follow-up rewrite failed (%s), using original.", exc)

    return question


# ──────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────

class GeminiEmptyResponseError(Exception):
    """Raised when Gemini returns an empty or blocked response."""
    pass
