#!/usr/bin/env python
"""
run_eval.py – Retrieval & generation evaluation for the YouTube RAG pipeline.

Usage
-----
    python -m eval.run_eval                           # default testset.json
    python -m eval.run_eval --testset eval/my_set.json
    python -m eval.run_eval --reranker                # enable reranker

Metrics
-------
* **Hit Rate@k** – fraction of queries where the ground-truth time range
  overlaps with at least one of the top-k retrieved chunks.
* **MRR** (Mean Reciprocal Rank) – average of 1/rank for the first hit.
* **Timestamp Accuracy** – fraction of queries where a cited chunk's start
  is within ±15 s of the ground-truth start.
* **Average Latency** – mean retrieval (+ optional rerank) time per query.

Results are broken down by ``question_language`` and ``transcript_source``.
A CSV is written alongside the test set, and a summary table is printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K, USE_RERANKER
from db.sqlite import get_db
from retrieval.embedder import get_embedder
from retrieval.store import SearchResult, get_store, retrieve, passes_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("eval")

# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class TestCase:
    """One entry from the test set JSON."""

    question: str
    question_language: str
    video_id: str
    ground_truth_start: float     # seconds
    ground_truth_end: float       # seconds


@dataclass
class EvalResult:
    """Per-query evaluation result."""

    question: str
    question_language: str
    video_id: str
    transcript_source: str        # manual | auto-generated
    hit: bool                     # ground truth overlaps a retrieved chunk
    rank: int                     # 1-based rank of first hit (0 = miss)
    timestamp_accurate: bool      # cited chunk within ±15s
    latency_ms: float
    best_score: float
    passed_threshold: bool


# ──────────────────────────────────────────────
# Evaluation logic
# ──────────────────────────────────────────────

TIMESTAMP_TOLERANCE = 15.0  # seconds


def _overlaps(
    chunk_start: float,
    chunk_end: float,
    gt_start: float,
    gt_end: float,
) -> bool:
    """True if the chunk time range overlaps the ground-truth range."""
    return chunk_start <= gt_end and chunk_end >= gt_start


def _within_tolerance(
    chunk_start: float,
    gt_start: float,
    tolerance: float = TIMESTAMP_TOLERANCE,
) -> bool:
    return abs(chunk_start - gt_start) <= tolerance


def evaluate_query(
    tc: TestCase,
    embedder: Any,
    store: Any,
    use_reranker: bool,
    top_k: int = RETRIEVAL_TOP_K,
    final_k: int = RETRIEVAL_FINAL_K,
) -> EvalResult:
    """Run retrieval for one test case and compute metrics."""
    db = get_db()
    video_rec = db.get_video(tc.video_id)
    transcript_source = video_rec.transcript_src if video_rec else "unknown"

    results, latency_ms = retrieve(
        tc.question,
        embedder,
        store,
        video_ids=[tc.video_id],
        use_reranker=use_reranker,
        top_k=top_k,
        final_k=final_k,
    )

    hit = False
    rank = 0
    ts_accurate = False

    for i, r in enumerate(results, 1):
        if _overlaps(r.start, r.end, tc.ground_truth_start, tc.ground_truth_end):
            if not hit:
                hit = True
                rank = i
            if _within_tolerance(r.start, tc.ground_truth_start):
                ts_accurate = True

    best_score = results[0].score if results else 0.0
    passed = passes_threshold(results, use_reranker)

    return EvalResult(
        question=tc.question,
        question_language=tc.question_language,
        video_id=tc.video_id,
        transcript_source=transcript_source,
        hit=hit,
        rank=rank,
        timestamp_accurate=ts_accurate,
        latency_ms=latency_ms,
        best_score=best_score,
        passed_threshold=passed,
    )


# ──────────────────────────────────────────────
# Aggregation & reporting
# ──────────────────────────────────────────────

def _aggregate(
    results: list[EvalResult],
    group_key: str,
) -> dict[str, dict[str, Any]]:
    """Group results by *group_key* and compute aggregate metrics."""
    groups: dict[str, list[EvalResult]] = {}
    for r in results:
        key = getattr(r, group_key)
        groups.setdefault(key, []).append(r)

    agg: dict[str, dict[str, Any]] = {}
    for key, items in sorted(groups.items()):
        n = len(items)
        hits = sum(1 for r in items if r.hit)
        mrr = sum((1.0 / r.rank) if r.rank > 0 else 0.0 for r in items) / n
        ts_acc = sum(1 for r in items if r.timestamp_accurate) / n
        avg_latency = sum(r.latency_ms for r in items) / n
        agg[key] = {
            "count": n,
            "hit_rate": hits / n,
            "mrr": mrr,
            "timestamp_accuracy": ts_acc,
            "avg_latency_ms": avg_latency,
        }
    return agg


def _print_table(title: str, agg: dict[str, dict[str, Any]]) -> None:
    """Pretty-print an aggregation table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = f"{'Group':<25} {'N':>4} {'Hit@k':>7} {'MRR':>7} {'TS±15s':>7} {'Lat(ms)':>9}"
    print(header)
    print("-" * len(header))
    for key, m in agg.items():
        print(
            f"{key:<25} {m['count']:>4} "
            f"{m['hit_rate']:>7.1%} {m['mrr']:>7.3f} "
            f"{m['timestamp_accuracy']:>7.1%} {m['avg_latency_ms']:>9.1f}"
        )
    # Overall
    all_items = [v for v in agg.values()]
    if all_items:
        total_n = sum(v["count"] for v in all_items)
        overall_hr = sum(v["hit_rate"] * v["count"] for v in all_items) / total_n
        overall_mrr = sum(v["mrr"] * v["count"] for v in all_items) / total_n
        overall_ts = sum(v["timestamp_accuracy"] * v["count"] for v in all_items) / total_n
        overall_lat = sum(v["avg_latency_ms"] * v["count"] for v in all_items) / total_n
        print("-" * len(header))
        print(
            f"{'OVERALL':<25} {total_n:>4} "
            f"{overall_hr:>7.1%} {overall_mrr:>7.3f} "
            f"{overall_ts:>7.1%} {overall_lat:>9.1f}"
        )


def _write_csv(results: list[EvalResult], path: Path) -> None:
    """Write per-query results to CSV."""
    fieldnames = [
        "question", "question_language", "video_id", "transcript_source",
        "hit", "rank", "timestamp_accurate", "latency_ms",
        "best_score", "passed_threshold",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question": r.question,
                "question_language": r.question_language,
                "video_id": r.video_id,
                "transcript_source": r.transcript_source,
                "hit": r.hit,
                "rank": r.rank,
                "timestamp_accurate": r.timestamp_accurate,
                "latency_ms": f"{r.latency_ms:.1f}",
                "best_score": f"{r.best_score:.4f}",
                "passed_threshold": r.passed_threshold,
            })
    print(f"\nCSV written to: {path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the YouTube RAG retrieval pipeline.",
    )
    parser.add_argument(
        "--testset", "-t",
        default=str(PROJECT_ROOT / "eval" / "testset.json"),
        help="Path to the test set JSON file.",
    )
    parser.add_argument(
        "--reranker", action="store_true",
        help="Enable the reranker for this run (overrides config.USE_RERANKER).",
    )
    parser.add_argument(
        "--top-k", type=int, default=RETRIEVAL_TOP_K,
        help="Number of initial vector-search candidates.",
    )
    parser.add_argument(
        "--final-k", type=int, default=RETRIEVAL_FINAL_K,
        help="Number of results to keep.",
    )
    args = parser.parse_args()

    use_reranker = args.reranker or USE_RERANKER

    # Load test set
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"ERROR: Test set not found: {testset_path}")
        print("Create a JSON file with the format shown in eval/testset.json")
        sys.exit(1)

    with open(testset_path, encoding="utf-8") as f:
        raw = json.load(f)

    test_cases = [
        TestCase(
            question=item["question"],
            question_language=item.get("question_language", "unknown"),
            video_id=item["video_id"],
            ground_truth_start=float(item["ground_truth_start"]),
            ground_truth_end=float(item["ground_truth_end"]),
        )
        for item in raw
    ]

    if not test_cases:
        print("ERROR: Test set is empty.")
        sys.exit(1)

    print(f"Loaded {len(test_cases)} test cases from {testset_path}")
    print(f"Reranker: {'ON' if use_reranker else 'OFF'}")
    print(f"top_k={args.top_k}, final_k={args.final_k}")

    # Load models
    embedder = get_embedder()
    store_inst = get_store()

    # Run evaluation
    eval_results: list[EvalResult] = []
    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc.question[:60]}…")
        result = evaluate_query(
            tc, embedder, store_inst,
            use_reranker=use_reranker,
            top_k=args.top_k,
            final_k=args.final_k,
        )
        eval_results.append(result)

    # Aggregate & print
    by_lang = _aggregate(eval_results, "question_language")
    _print_table("By Question Language", by_lang)

    by_src = _aggregate(eval_results, "transcript_source")
    _print_table("By Transcript Source", by_src)

    # CSV output
    csv_name = f"eval_results_{'reranker' if use_reranker else 'vector'}.csv"
    csv_path = testset_path.parent / csv_name
    _write_csv(eval_results, csv_path)


if __name__ == "__main__":
    main()
