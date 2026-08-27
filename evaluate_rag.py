from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import time
from pathlib import Path
from typing import Any

from chatbot.chain import (
    ask_question,
    get_llm,
)

from chatbot.retriever import (
    get_embedding_model,
)

from config import DEFAULT_MODEL


LOGGER = logging.getLogger(
    __name__
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_INPUT_PATH = (
    "new_evaluation_dataset.csv"
)

DEFAULT_RESULTS_PATH = (
    "evaluation_results.csv"
)

DEFAULT_SUMMARY_PATH = (
    "ragas_summary.csv"
)

EVAL_USER_ID = "ragas_test"

# Fixed RAGAS evaluator.
# This stays the same when benchmarking Mistral or OxAlpha.
RAGAS_EVALUATOR_MODEL = "mistral-small-latest"


RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


PERFORMANCE_METRICS = [
    "average_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "average_retrieval_latency_ms",
    "p50_retrieval_latency_ms",
    "p95_retrieval_latency_ms",
    "p99_retrieval_latency_ms",
    "average_context_expansion_latency_ms",
    "p50_context_expansion_latency_ms",
    "p95_context_expansion_latency_ms",
    "p99_context_expansion_latency_ms",
    "average_memory_retrieval_latency_ms",
    "p50_memory_retrieval_latency_ms",
    "p95_memory_retrieval_latency_ms",
    "p99_memory_retrieval_latency_ms",
    "average_llm_latency_ms",
    "p50_llm_latency_ms",
    "p95_llm_latency_ms",
    "p99_llm_latency_ms",
    "average_memory_write_latency_ms",
    "p50_memory_write_latency_ms",
    "p95_memory_write_latency_ms",
    "p99_memory_write_latency_ms",
    "average_input_tokens",
    "average_output_tokens",
    "average_total_tokens",
    "average_tokens_per_second",
    "success_rate",
    "error_rate",
]


RESULT_COLUMNS = [
    "question",
    "ground_truth",
    "answer",
    "retrieved_contexts",
    "retrieved_sources",
    "retrieved_pages",
    "status",
    "guardrail_reason",
    "retrieved_count",
    "context_count",
    "memory_count",

    # RAGAS
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",

    # Performance
    "latency_ms",
    "retrieval_latency_ms",
    "context_expansion_latency_ms",
    "memory_retrieval_latency_ms",
    "llm_latency_ms",
    "memory_write_latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tokens_per_second",
]


# ============================================================================
# Logging
# ============================================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s - %(message)s"
        ),
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the School Chatbot "
            "RAG pipeline."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--results-output",
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--summary-output",
        default=DEFAULT_SUMMARY_PATH,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "LLM model used by the chatbot "
            "being benchmarked."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Number of questions to evaluate. "
            "0 means all."
        ),
    )

    parser.add_argument(
        "--ragas-only",
        action="store_true",
        help=(
            "Run RAGAS only on an existing "
            "evaluation results CSV. "
            "Does not call the chatbot."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Dataset
# ============================================================================

def load_eval_rows(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        fieldnames = set(
            reader.fieldnames or []
        )

        required = {
            "question",
            "ground_truth",
        }

        missing = (
            required - fieldnames
        )

        if missing:
            raise ValueError(
                "Evaluation CSV is missing "
                f"required columns: "
                f"{sorted(missing)}"
            )

        rows = []

        for row in reader:

            question = (
                row.get(
                    "question"
                )
                or ""
            ).strip()

            ground_truth = (
                row.get(
                    "ground_truth"
                )
                or ""
            ).strip()

            if not question:

                LOGGER.warning(
                    "Skipping empty question."
                )

                continue

            rows.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )

        return rows


def load_existing_results(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Existing results file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        rows = []

        for row in reader:

            normalized_row = {
                key: (
                    value
                    if value is not None
                    else ""
                )
                for key, value in row.items()
            }

            rows.append(
                normalized_row
            )

    LOGGER.info(
        "Loaded %d existing evaluation "
        "results from %s",
        len(rows),
        path,
    )

    return rows


# ============================================================================
# Document helpers
# ============================================================================

def doc_to_context(
    doc: Any,
) -> str:

    return str(
        getattr(
            doc,
            "page_content",
            "",
        )
    ).strip()


def doc_to_source(
    doc: Any,
) -> str:

    metadata = getattr(
        doc,
        "metadata",
        {},
    ) or {}

    return str(
        metadata.get(
            "source",
            "Unknown",
        )
    )


def doc_to_page(
    doc: Any,
) -> str:

    metadata = getattr(
        doc,
        "metadata",
        {},
    ) or {}

    return str(
        metadata.get(
            "page_number",
            metadata.get(
                "page",
                "Unknown",
            ),
        )
    )


# ============================================================================
# Per-question evaluation
# ============================================================================

def run_sample(
    question: str,
    ground_truth: str,
    model_name: str,
) -> dict[str, str]:

    row = {
        "question": question,
        "ground_truth": ground_truth,
        "answer": "",

        "retrieved_contexts": json.dumps(
            [],
            ensure_ascii=False,
        ),

        "retrieved_sources": json.dumps(
            [],
            ensure_ascii=False,
        ),

        "retrieved_pages": json.dumps(
            [],
            ensure_ascii=False,
        ),

        "status": "failed",
        "guardrail_reason": "",

        "retrieved_count": "0",
        "context_count": "0",
        "memory_count": "0",

        # RAGAS
        "faithfulness": "",
        "answer_relevancy": "",
        "context_precision": "",
        "context_recall": "",

        # Performance
        "latency_ms": "",
        "retrieval_latency_ms": "",
        "context_expansion_latency_ms": "",
        "memory_retrieval_latency_ms": "",
        "llm_latency_ms": "",
        "memory_write_latency_ms": "",
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
        "tokens_per_second": "",
    }

    start_time = time.perf_counter()

    try:

        answer, docs, metadata = (
            ask_question(
                question=question,
                model_name=model_name,
                chat_history=[],
                user_id=EVAL_USER_ID,
                return_metadata=True,
            )
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        # ------------------------------------------------------------
        # Latency
        # ------------------------------------------------------------

        row["latency_ms"] = (
            f"{elapsed_seconds * 1000:.3f}"
        )

        # ------------------------------------------------------------
        # Basic metadata
        # ------------------------------------------------------------

        row["status"] = str(
            metadata.get(
                "status",
                "answered",
            )
            or "answered"
        ).strip()

        row["guardrail_reason"] = str(
            metadata.get(
                "guardrail_reason",
                "",
            )
            or ""
        ).strip()

        row["retrieved_count"] = str(
            metadata.get(
                "retrieved_count",
                0,
            )
        )

        row["context_count"] = str(
            metadata.get(
                "context_count",
                len(docs),
            )
        )

        row["memory_count"] = str(
            metadata.get(
                "memory_count",
                0,
            )
        )

        # ------------------------------------------------------------
        # Component latency
        # ------------------------------------------------------------

        latency_fields = [
            "retrieval_latency_ms",
            "context_expansion_latency_ms",
            "memory_retrieval_latency_ms",
            "llm_latency_ms",
            "memory_write_latency_ms",
        ]

        for field in latency_fields:

            value = metadata.get(
                field
            )

            if value is not None:

                row[field] = (
                    f"{float(value):.3f}"
                )

        # ------------------------------------------------------------
        # Token usage
        # ------------------------------------------------------------

        input_tokens = metadata.get(
            "input_tokens"
        )

        output_tokens = metadata.get(
            "output_tokens"
        )

        total_tokens = metadata.get(
            "total_tokens"
        )

        if input_tokens is not None:

            row["input_tokens"] = str(
                input_tokens
            )

        if output_tokens is not None:

            row["output_tokens"] = str(
                output_tokens
            )

        if total_tokens is not None:

            row["total_tokens"] = str(
                total_tokens
            )

        if (
            output_tokens is not None
            and elapsed_seconds > 0
        ):

            row["tokens_per_second"] = (
                f"{float(output_tokens) / elapsed_seconds:.3f}"
            )

        # ------------------------------------------------------------
        # Guardrail blocked
        # ------------------------------------------------------------

        if row["status"] == "blocked":

            return row

        # ------------------------------------------------------------
        # Answer
        # ------------------------------------------------------------

        row["answer"] = str(
            answer or ""
        ).strip()

        # ------------------------------------------------------------
        # Retrieved contexts
        # ------------------------------------------------------------

        contexts = []
        sources = []
        pages = []

        for doc in docs:

            context = (
                doc_to_context(
                    doc
                )
            )

            if not context:
                continue

            contexts.append(
                context
            )

            sources.append(
                doc_to_source(
                    doc
                )
            )

            pages.append(
                doc_to_page(
                    doc
                )
            )

        row[
            "retrieved_contexts"
        ] = json.dumps(
            contexts,
            ensure_ascii=False,
        )

        row[
            "retrieved_sources"
        ] = json.dumps(
            sources,
            ensure_ascii=False,
        )

        row[
            "retrieved_pages"
        ] = json.dumps(
            pages,
            ensure_ascii=False,
        )

        row["status"] = "answered"

        return row

    except Exception:

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        row["latency_ms"] = (
            f"{elapsed_seconds * 1000:.3f}"
        )

        LOGGER.exception(
            "Evaluation sample failed: %s",
            question,
        )

        return row


# ============================================================================
# RAGAS records
# ============================================================================

def parse_contexts(
    value: str,
) -> list[str]:

    if not value:
        return []

    try:

        parsed = json.loads(
            value
        )

    except json.JSONDecodeError:

        return []

    if not isinstance(
        parsed,
        list,
    ):

        return []

    return [
        str(item).strip()
        for item in parsed
        if str(item).strip()
    ]


def build_ragas_records(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[int],
]:

    records = []
    answered_indexes = []

    for index, row in enumerate(
        rows
    ):

        if row.get(
            "status"
        ) != "answered":

            continue

        question = (
            row.get(
                "question"
            )
            or ""
        ).strip()

        answer = (
            row.get(
                "answer"
            )
            or ""
        ).strip()

        ground_truth = (
            row.get(
                "ground_truth"
            )
            or ""
        ).strip()

        contexts = parse_contexts(
            row.get(
                "retrieved_contexts",
                "",
            )
        )

        if not question:
            continue

        if not answer:
            continue

        records.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": ground_truth,

                # Compatibility
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth,
            }
        )

        answered_indexes.append(
            index
        )

    return (
        records,
        answered_indexes,
    )


# ============================================================================
# RAGAS
# ============================================================================

def build_metrics():

    from ragas.metrics import (
        ContextPrecision,
        Faithfulness,
        LLMContextRecall,
        ResponseRelevancy,
    )

    return [
        Faithfulness(),
        ResponseRelevancy(),
        ContextPrecision(),
        LLMContextRecall(),
    ]


def run_ragas(
    records: list[dict[str, Any]],
):

    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import (
        LangchainLLMWrapper,
    )
    from ragas.run_config import (
        RunConfig,
    )

    dataset = Dataset.from_list(
        records
    )

    # IMPORTANT:
    # RAGAS always uses the same fixed
    # evaluator model.
    evaluator_llm = (
        LangchainLLMWrapper(
            get_llm(
                RAGAS_EVALUATOR_MODEL,
                max_tokens=4096,
            )
        )
    )

    evaluator_embeddings = (
        get_embedding_model()
    )

    # Lower concurrency to reduce
    # temporary 503 errors from the
    # evaluator endpoint.
    run_config = RunConfig(
        timeout=300,
        max_workers=1,
        max_retries=5,
    )

    LOGGER.info(
        "Running RAGAS on %d answered samples.",
        len(records),
    )

    return evaluate(
        dataset=dataset,
        metrics=build_metrics(),
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        show_progress=True,
        run_config=run_config,
    )


# ============================================================================
# RAGAS result extraction
# ============================================================================

def normalize_metric_value(
    value: Any,
) -> float | None:

    if isinstance(
        value,
        (int, float),
    ):

        numeric = float(
            value
        )

        if not math.isnan(
            numeric
        ):

            return numeric

    return None


def extract_score_rows(
    result: Any,
) -> list[dict[str, Any]]:

    scores = getattr(
        result,
        "scores",
        None,
    )

    if isinstance(
        scores,
        list,
    ):

        return [
            row
            for row in scores
            if isinstance(
                row,
                dict,
            )
        ]

    to_pandas = getattr(
        result,
        "to_pandas",
        None,
    )

    if callable(
        to_pandas
    ):

        dataframe = (
            to_pandas()
        )

        return dataframe.to_dict(
            orient="records"
        )

    return []


def merge_ragas_scores(
    rows: list[dict[str, str]],
    answered_indexes: list[int],
    score_rows: list[dict[str, Any]],
) -> None:

    for row in rows:

        for metric in (
            RAGAS_METRICS
        ):

            row[
                metric
            ] = row.get(
                metric,
                "",
            )

    for row_index, score_row in zip(
        answered_indexes,
        score_rows,
    ):

        target_row = rows[
            row_index
        ]

        for metric in (
            RAGAS_METRICS
        ):

            value = (
                normalize_metric_value(
                    score_row.get(
                        metric
                    )
                )
            )

            target_row[
                metric
            ] = (
                ""
                if value is None
                else f"{value:.6f}"
            )


# ============================================================================
# Numeric helpers
# ============================================================================

def numeric_values(
    rows: list[dict[str, str]],
    field: str,
) -> list[float]:

    values = []

    for row in rows:

        value = row.get(
            field,
            "",
        )

        if value in (
            "",
            None,
        ):
            continue

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        if not math.isnan(
            number
        ):

            values.append(
                number
            )

    return values


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        return float("nan")

    values = sorted(
        values
    )

    if len(values) == 1:
        return values[0]

    position = (
        (percentile_value / 100)
        * (len(values) - 1)
    )

    lower = math.floor(
        position
    )

    upper = math.ceil(
        position
    )

    if lower == upper:

        return values[
            lower
        ]

    weight = (
        position - lower
    )

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * weight
    )


# ============================================================================
# Aggregation
# ============================================================================

def compute_ragas_averages(
    rows: list[dict[str, str]],
) -> dict[str, float]:

    result = {}

    for metric in (
        RAGAS_METRICS
    ):

        values = numeric_values(
            rows,
            metric,
        )

        result[
            metric
        ] = (
            statistics.mean(
                values
            )
            if values
            else float("nan")
        )

    return result


def compute_performance(
    rows: list[dict[str, str]],
) -> dict[str, float]:

    latency = numeric_values(
        rows,
        "latency_ms",
    )

    retrieval_latency = numeric_values(
        rows,
        "retrieval_latency_ms",
    )

    context_expansion_latency = numeric_values(
        rows,
        "context_expansion_latency_ms",
    )

    memory_retrieval_latency = numeric_values(
        rows,
        "memory_retrieval_latency_ms",
    )

    llm_latency = numeric_values(
        rows,
        "llm_latency_ms",
    )

    memory_write_latency = numeric_values(
        rows,
        "memory_write_latency_ms",
    )

    input_tokens = numeric_values(
        rows,
        "input_tokens",
    )

    output_tokens = numeric_values(
        rows,
        "output_tokens",
    )

    total_tokens = numeric_values(
        rows,
        "total_tokens",
    )

    tokens_per_second = numeric_values(
        rows,
        "tokens_per_second",
    )

    total = len(
        rows
    )

    answered = sum(
        1
        for row in rows
        if row.get(
            "status"
        ) == "answered"
    )

    failed = sum(
        1
        for row in rows
        if row.get(
            "status"
        ) == "failed"
    )

    return {
        "average_latency_ms": (
            statistics.mean(
                latency
            )
            if latency
            else float("nan")
        ),

        "p50_latency_ms": percentile(
            latency,
            50,
        ),

        "p95_latency_ms": percentile(
            latency,
            95,
        ),

        "p99_latency_ms": percentile(
            latency,
            99,
        ),

        "average_retrieval_latency_ms": (
            statistics.mean(
                retrieval_latency
            )
            if retrieval_latency
            else float("nan")
        ),

        "p50_retrieval_latency_ms": percentile(
            retrieval_latency,
            50,
        ),

        "p95_retrieval_latency_ms": percentile(
            retrieval_latency,
            95,
        ),

        "p99_retrieval_latency_ms": percentile(
            retrieval_latency,
            99,
        ),

        "average_context_expansion_latency_ms": (
            statistics.mean(
                context_expansion_latency
            )
            if context_expansion_latency
            else float("nan")
        ),

        "p50_context_expansion_latency_ms": percentile(
            context_expansion_latency,
            50,
        ),

        "p95_context_expansion_latency_ms": percentile(
            context_expansion_latency,
            95,
        ),

        "p99_context_expansion_latency_ms": percentile(
            context_expansion_latency,
            99,
        ),

        "average_memory_retrieval_latency_ms": (
            statistics.mean(
                memory_retrieval_latency
            )
            if memory_retrieval_latency
            else float("nan")
        ),

        "p50_memory_retrieval_latency_ms": percentile(
            memory_retrieval_latency,
            50,
        ),

        "p95_memory_retrieval_latency_ms": percentile(
            memory_retrieval_latency,
            95,
        ),

        "p99_memory_retrieval_latency_ms": percentile(
            memory_retrieval_latency,
            99,
        ),

        "average_llm_latency_ms": (
            statistics.mean(
                llm_latency
            )
            if llm_latency
            else float("nan")
        ),

        "p50_llm_latency_ms": percentile(
            llm_latency,
            50,
        ),

        "p95_llm_latency_ms": percentile(
            llm_latency,
            95,
        ),

        "p99_llm_latency_ms": percentile(
            llm_latency,
            99,
        ),

        "average_memory_write_latency_ms": (
            statistics.mean(
                memory_write_latency
            )
            if memory_write_latency
            else float("nan")
        ),

        "p50_memory_write_latency_ms": percentile(
            memory_write_latency,
            50,
        ),

        "p95_memory_write_latency_ms": percentile(
            memory_write_latency,
            95,
        ),

        "p99_memory_write_latency_ms": percentile(
            memory_write_latency,
            99,
        ),

        "average_input_tokens": (
            statistics.mean(
                input_tokens
            )
            if input_tokens
            else float("nan")
        ),

        "average_output_tokens": (
            statistics.mean(
                output_tokens
            )
            if output_tokens
            else float("nan")
        ),

        "average_total_tokens": (
            statistics.mean(
                total_tokens
            )
            if total_tokens
            else float("nan")
        ),

        "average_tokens_per_second": (
            statistics.mean(
                tokens_per_second
            )
            if tokens_per_second
            else float("nan")
        ),

        "success_rate": (
            answered / total
            if total
            else float("nan")
        ),

        "error_rate": (
            failed / total
            if total
            else float("nan")
        ),
    }


# ============================================================================
# Output
# ============================================================================

def save_results(
    path: Path,
    rows: list[dict[str, str]],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=RESULT_COLUMNS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def save_summary(
    path: Path,
    ragas_metrics: dict[str, float],
    performance_metrics: dict[str, float],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "metric",
                "score",
            ],
        )

        writer.writeheader()

        for metric in (
            RAGAS_METRICS
        ):

            value = (
                ragas_metrics.get(
                    metric,
                    float("nan"),
                )
            )

            writer.writerow(
                {
                    "metric": metric,
                    "score": (
                        ""
                        if math.isnan(
                            value
                        )
                        else f"{value:.6f}"
                    ),
                }
            )

        for metric in (
            PERFORMANCE_METRICS
        ):

            value = (
                performance_metrics.get(
                    metric,
                    float("nan"),
                )
            )

            writer.writerow(
                {
                    "metric": metric,
                    "score": (
                        ""
                        if math.isnan(
                            value
                        )
                        else f"{value:.6f}"
                    ),
                }
            )


# ============================================================================
# Counts
# ============================================================================

def summarize_counts(
    rows: list[dict[str, str]],
) -> dict[str, int]:

    return {
        "total": len(
            rows
        ),

        "answered": sum(
            1
            for row in rows
            if row.get(
                "status"
            ) == "answered"
        ),

        "blocked": sum(
            1
            for row in rows
            if row.get(
                "status"
            ) == "blocked"
        ),

        "failed": sum(
            1
            for row in rows
            if row.get(
                "status"
            ) == "failed"
        ),
    }


def format_value(
    value: float,
) -> str:

    if math.isnan(
        value
    ):

        return "N/A"

    return f"{value:.6f}"


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    setup_logging()

    args = parse_args()

    input_path = Path(
        args.input
    )

    results_path = Path(
        args.results_output
    )

    summary_path = Path(
        args.summary_output
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    if args.ragas_only:

        rows = load_existing_results(
            results_path
        )

    else:

        rows = load_eval_rows(
            input_path
        )

    if args.limit > 0:

        rows = rows[
            :args.limit
        ]

    LOGGER.info(
        "Loaded %d evaluation rows.",
        len(rows),
    )

    results = []

    total = len(
        rows
    )

    # ------------------------------------------------------------------
    # Run chatbot
    # ------------------------------------------------------------------

    if args.ragas_only:

        # IMPORTANT:
        # Reuse existing chatbot results.
        # No chatbot calls happen here.
        results = rows

        LOGGER.info(
            "RAGAS-only mode enabled. "
            "Skipping chatbot generation."
        )

    else:

        for index, row in enumerate(
            rows,
            start=1,
        ):

            print(
                f"[{index}/{total}] "
                "Processing question"
            )

            results.append(
                run_sample(
                    question=row[
                        "question"
                    ],
                    ground_truth=row[
                        "ground_truth"
                    ],
                    model_name=args.model,
                )
            )

    # ------------------------------------------------------------------
    # RAGAS
    # ------------------------------------------------------------------

    records, answered_indexes = (
        build_ragas_records(
            results
        )
    )

    LOGGER.info(
        "Prepared %d answered rows for RAGAS.",
        len(records),
    )

    if records:

        try:

            ragas_result = (
                run_ragas(
                    records,
                )
            )

            print(
                "=" * 80
            )

            print(
                "RAGAS RESULT"
            )

            print(
                ragas_result
            )

            print(
                "=" * 80
            )

            score_rows = (
                extract_score_rows(
                    ragas_result
                )
            )

            print(
                "Score rows:",
                len(score_rows),
            )

            if score_rows:

                print(
                    "First score row:"
                )

                print(
                    score_rows[0]
                )

            merge_ragas_scores(
                results,
                answered_indexes,
                score_rows,
            )

        except Exception:

            LOGGER.exception(
                "RAGAS evaluation failed."
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    ragas_metrics = (
        compute_ragas_averages(
            results
        )
    )

    performance_metrics = (
        compute_performance(
            results
        )
    )

    counts = summarize_counts(
        results
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    # Do NOT overwrite the original
    # chatbot results in RAGAS-only mode.
    if not args.ragas_only:

        save_results(
            results_path,
            results,
        )

    save_summary(
        summary_path,
        ragas_metrics,
        performance_metrics,
    )

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )

    if args.ragas_only:

        print(
            "Mode: RAGAS-only"
        )

        print(
            "Source: "
            f"{results_path}"
        )

        print(
            "Chatbot model: "
            "x-preview-f-free"
        )

        print(
            "RAGAS evaluator: "
            f"{RAGAS_EVALUATOR_MODEL}"
        )

    else:

        print(
            f"Model: {args.model}"
        )

    print(
        f"Total questions: "
        f"{counts['total']}"
    )

    print(
        f"Answered: "
        f"{counts['answered']}"
    )

    print(
        f"Blocked: "
        f"{counts['blocked']}"
    )

    print(
        f"Failed: "
        f"{counts['failed']}"
    )

    print()

    print(
        "RAGAS QUALITY"
    )

    print(
        f"Average Faithfulness: "
        f"{format_value(ragas_metrics['faithfulness'])}"
    )

    print(
        f"Average Answer Relevancy: "
        f"{format_value(ragas_metrics['answer_relevancy'])}"
    )

    print(
        f"Average Context Precision: "
        f"{format_value(ragas_metrics['context_precision'])}"
    )

    print(
        f"Average Context Recall: "
        f"{format_value(ragas_metrics['context_recall'])}"
    )

    print()

    print(
        "PERFORMANCE"
    )

    print(
        f"Average Latency: "
        f"{format_value(performance_metrics['average_latency_ms'])} ms"
    )

    print(
        f"P50 Latency: "
        f"{format_value(performance_metrics['p50_latency_ms'])} ms"
    )

    print(
        f"P95 Latency: "
        f"{format_value(performance_metrics['p95_latency_ms'])} ms"
    )

    print(
        f"P99 Latency: "
        f"{format_value(performance_metrics['p99_latency_ms'])} ms"
    )

    print()

    print(
        "RETRIEVAL LATENCY"
    )

    print(
        f"Average Retrieval Latency: "
        f"{format_value(performance_metrics['average_retrieval_latency_ms'])} ms"
    )

    print(
        f"P50 Retrieval Latency: "
        f"{format_value(performance_metrics['p50_retrieval_latency_ms'])} ms"
    )

    print(
        f"P95 Retrieval Latency: "
        f"{format_value(performance_metrics['p95_retrieval_latency_ms'])} ms"
    )

    print(
        f"P99 Retrieval Latency: "
        f"{format_value(performance_metrics['p99_retrieval_latency_ms'])} ms"
    )

    print()

    print(
        "CONTEXT EXPANSION LATENCY"
    )

    print(
        f"Average Context Expansion Latency: "
        f"{format_value(performance_metrics['average_context_expansion_latency_ms'])} ms"
    )

    print(
        f"P50 Context Expansion Latency: "
        f"{format_value(performance_metrics['p50_context_expansion_latency_ms'])} ms"
    )

    print(
        f"P95 Context Expansion Latency: "
        f"{format_value(performance_metrics['p95_context_expansion_latency_ms'])} ms"
    )

    print(
        f"P99 Context Expansion Latency: "
        f"{format_value(performance_metrics['p99_context_expansion_latency_ms'])} ms"
    )

    print()

    print(
        "MEMORY RETRIEVAL LATENCY"
    )

    print(
        f"Average Memory Retrieval Latency: "
        f"{format_value(performance_metrics['average_memory_retrieval_latency_ms'])} ms"
    )

    print(
        f"P50 Memory Retrieval Latency: "
        f"{format_value(performance_metrics['p50_memory_retrieval_latency_ms'])} ms"
    )

    print(
        f"P95 Memory Retrieval Latency: "
        f"{format_value(performance_metrics['p95_memory_retrieval_latency_ms'])} ms"
    )

    print(
        f"P99 Memory Retrieval Latency: "
        f"{format_value(performance_metrics['p99_memory_retrieval_latency_ms'])} ms"
    )

    print()

    print(
        "LLM LATENCY"
    )

    print(
        f"Average LLM Latency: "
        f"{format_value(performance_metrics['average_llm_latency_ms'])} ms"
    )

    print(
        f"P50 LLM Latency: "
        f"{format_value(performance_metrics['p50_llm_latency_ms'])} ms"
    )

    print(
        f"P95 LLM Latency: "
        f"{format_value(performance_metrics['p95_llm_latency_ms'])} ms"
    )

    print(
        f"P99 LLM Latency: "
        f"{format_value(performance_metrics['p99_llm_latency_ms'])} ms"
    )

    print()

    print(
        "MEMORY WRITE LATENCY"
    )

    print(
        f"Average Memory Write Latency: "
        f"{format_value(performance_metrics['average_memory_write_latency_ms'])} ms"
    )

    print(
        f"P50 Memory Write Latency: "
        f"{format_value(performance_metrics['p50_memory_write_latency_ms'])} ms"
    )

    print(
        f"P95 Memory Write Latency: "
        f"{format_value(performance_metrics['p95_memory_write_latency_ms'])} ms"
    )

    print(
        f"P99 Memory Write Latency: "
        f"{format_value(performance_metrics['p99_memory_write_latency_ms'])} ms"
    )

    print()

    print(
        f"Average Input Tokens: "
        f"{format_value(performance_metrics['average_input_tokens'])}"
    )

    print(
        f"Average Output Tokens: "
        f"{format_value(performance_metrics['average_output_tokens'])}"
    )

    print(
        f"Average Total Tokens: "
        f"{format_value(performance_metrics['average_total_tokens'])}"
    )

    print(
        f"Average Tokens/sec: "
        f"{format_value(performance_metrics['average_tokens_per_second'])}"
    )

    print(
        f"Success Rate: "
        f"{format_value(performance_metrics['success_rate'])}"
    )

    print(
        f"Error Rate: "
        f"{format_value(performance_metrics['error_rate'])}"
    )

    print()

    if args.ragas_only:

        print(
            "Original evaluation_results.csv "
            "was NOT overwritten."
        )

    else:

        print(
            f"Saved results to: "
            f"{results_path}"
        )

    print(
        f"Saved summary to: "
        f"{summary_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()