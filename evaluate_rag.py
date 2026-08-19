from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from chatbot.chain import ask_question, get_llm
from chatbot.rag_service import DEFAULT_MODEL
from chatbot.retriever import get_embedding_model

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = "new_evaluation_dataset.csv"
DEFAULT_RESULTS_PATH = "evaluation_results.csv"
DEFAULT_SUMMARY_PATH = "ragas_summary.csv"

EVAL_USER_ID = "ragas_test"

REQUESTED_METRIC_ORDER = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
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
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s - %(message)s"
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the School Chatbot production RAG pipeline "
            "and evaluate it with RAGAS."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=(
            "Evaluation CSV containing "
            "question,ground_truth."
        ),
    )

    parser.add_argument(
        "--results-output",
        default=DEFAULT_RESULTS_PATH,
        help=(
            "Per-question evaluation output CSV."
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=DEFAULT_SUMMARY_PATH,
        help=(
            "Aggregate RAGAS summary CSV."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model used for chatbot generation "
            "and RAGAS evaluation."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Evaluate only the first N questions. "
            "0 means all questions."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
        reader = csv.DictReader(csv_file)

        fieldnames = set(
            reader.fieldnames or []
        )

        missing = {
            "question",
            "ground_truth",
        } - fieldnames

        if missing:
            raise ValueError(
                "Evaluation CSV is missing required "
                f"columns: {sorted(missing)}"
            )

        rows: list[dict[str, str]] = []

        for row in reader:
            question = (
                row.get("question")
                or ""
            ).strip()

            ground_truth = (
                row.get("ground_truth")
                or ""
            ).strip()

            if not question:
                LOGGER.warning(
                    "Skipping empty evaluation question."
                )
                continue

            rows.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )

        return rows


# ---------------------------------------------------------------------------
# RAG result extraction
# ---------------------------------------------------------------------------

def doc_to_context(
    doc: Any,
) -> str:
    page_content = getattr(
        doc,
        "page_content",
        "",
    )

    return str(
        page_content
    ).strip()


def doc_to_source(
    doc: Any,
) -> str:
    metadata = getattr(
        doc,
        "metadata",
        {},
    )

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
    )

    page = metadata.get(
        "page_number",
        metadata.get(
            "page",
            "Unknown",
        ),
    )

    return str(page)


def empty_metric_fields() -> dict[str, str]:
    return {
        metric: ""
        for metric in REQUESTED_METRIC_ORDER
    }


# ---------------------------------------------------------------------------
# Per-question execution
# ---------------------------------------------------------------------------

def run_sample(
    question: str,
    ground_truth: str,
    model_name: str,
) -> dict[str, str]:

    base_row = {
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
        **empty_metric_fields(),
    }

    # Every evaluation sample gets a fresh conversation ID.
    # This prevents accidental conversation-state coupling.
    evaluation_conversation_id = (
        f"ragas-{uuid4().hex}"
    )

    try:
        answer, docs, metadata = ask_question(
            question=question,
            model_name=model_name,
            chat_history=[],
            user_id=EVAL_USER_ID,
            conversation_id=(
                evaluation_conversation_id
            ),
            return_metadata=True,
        )

        status = str(
            metadata.get(
                "status",
                "answered",
            )
            or "answered"
        ).strip()

        guardrail_reason = str(
            metadata.get(
                "guardrail_reason",
                "",
            )
            or ""
        ).strip()

        base_row["status"] = status
        base_row[
            "guardrail_reason"
        ] = guardrail_reason

        base_row["retrieved_count"] = str(
            metadata.get(
                "retrieved_count",
                0,
            )
        )

        base_row["context_count"] = str(
            metadata.get(
                "context_count",
                len(docs),
            )
        )

        base_row["memory_count"] = str(
            metadata.get(
                "memory_count",
                0,
            )
        )

        if status == "blocked":
            return base_row

        contexts = []
        sources = []
        pages = []

        for doc in docs:
            context = doc_to_context(doc)

            if not context:
                continue

            contexts.append(
                context
            )

            sources.append(
                doc_to_source(doc)
            )

            pages.append(
                doc_to_page(doc)
            )

        base_row["answer"] = str(
            answer or ""
        ).strip()

        base_row[
            "retrieved_contexts"
        ] = json.dumps(
            contexts,
            ensure_ascii=False,
        )

        base_row[
            "retrieved_sources"
        ] = json.dumps(
            sources,
            ensure_ascii=False,
        )

        base_row[
            "retrieved_pages"
        ] = json.dumps(
            pages,
            ensure_ascii=False,
        )

        base_row["status"] = "answered"

        return base_row

    except Exception:
        LOGGER.exception(
            "Evaluation sample failed: %s",
            question,
        )

        return base_row


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def parse_contexts(
    value: str,
) -> list[str]:
    if not value:
        return []

    try:
        parsed = json.loads(value)

    except json.JSONDecodeError:
        LOGGER.warning(
            "Could not parse retrieved_contexts JSON."
        )
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


# ---------------------------------------------------------------------------
# RAGAS dataset
# ---------------------------------------------------------------------------

def build_ragas_records(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[int],
]:
    records: list[
        dict[str, Any]
    ] = []

    answered_indexes: list[int] = []

    for index, row in enumerate(rows):
        if row.get("status") != "answered":
            continue

        question = (
            row.get("question")
            or ""
        ).strip()

        answer = (
            row.get("answer")
            or ""
        ).strip()

        ground_truth = (
            row.get("ground_truth")
            or ""
        ).strip()

        contexts = parse_contexts(
            row.get(
                "retrieved_contexts",
                "",
            )
        )

        if not question:
            LOGGER.warning(
                "Skipping row %d: empty question.",
                index,
            )
            continue

        if not answer:
            LOGGER.warning(
                "Skipping row %d: empty answer.",
                index,
            )
            continue

        if not contexts:
            LOGGER.warning(
                "Row %d has no retrieved contexts.",
                index,
            )

        records.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": ground_truth,

                # Compatibility aliases.
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


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------

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
    evaluator_model: str,
):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import (
            LangchainLLMWrapper,
        )
        from ragas.run_config import (
            RunConfig,
        )

    except ImportError as exc:
        raise ImportError(
            "Missing RAGAS evaluation dependencies. "
            "Install ragas and datasets."
        ) from exc

    dataset = Dataset.from_list(
        records
    )

    print("=" * 80)
    print("FIRST RECORD SENT TO RAGAS")
    print(
        json.dumps(
            records[0],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 80)

    evaluator_llm = (
        LangchainLLMWrapper(
            get_llm(
                evaluator_model,
                max_tokens=4096,
            )
        )
    )

    evaluator_embeddings = (
        get_embedding_model()
    )

    run_config = RunConfig(
        timeout=300,
        max_workers=2,
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
        raise_exceptions=True,
        show_progress=True,
        run_config=run_config,
    )


# ---------------------------------------------------------------------------
# RAGAS results
# ---------------------------------------------------------------------------

def normalize_metric_value(
    value: Any,
) -> float | None:
    if isinstance(
        value,
        (int, float),
    ):
        numeric = float(value)

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
        for metric in REQUESTED_METRIC_ORDER:
            row[metric] = row.get(
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

        for metric in REQUESTED_METRIC_ORDER:
            metric_value = (
                normalize_metric_value(
                    score_row.get(
                        metric
                    )
                )
            )

            target_row[metric] = (
                ""
                if metric_value is None
                else f"{metric_value:.6f}"
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_metric_averages(
    rows: list[dict[str, str]],
) -> dict[str, float]:
    averages: dict[
        str,
        float,
    ] = {}

    for metric in REQUESTED_METRIC_ORDER:
        values: list[float] = []

        for row in rows:
            raw_value = row.get(
                metric,
                "",
            )

            if raw_value in (
                "",
                None,
            ):
                continue

            try:
                numeric = float(
                    raw_value
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if not math.isnan(
                numeric
            ):
                values.append(
                    numeric
                )

        averages[metric] = (
            sum(values)
            / len(values)
            if values
            else float("nan")
        )

    return averages


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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
        writer.writerows(rows)


def save_summary(
    path: Path,
    averages: dict[str, float],
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

        for metric in REQUESTED_METRIC_ORDER:
            value = averages.get(
                metric,
                float("nan"),
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


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

def summarize_counts(
    rows: list[dict[str, str]],
) -> dict[str, int]:
    return {
        "total": len(rows),
        "answered": sum(
            1
            for row in rows
            if row.get(
                "status"
            )
            == "answered"
        ),
        "blocked": sum(
            1
            for row in rows
            if row.get(
                "status"
            )
            == "blocked"
        ),
        "failed": sum(
            1
            for row in rows
            if row.get(
                "status"
            )
            == "failed"
        ),
    }


def format_summary_value(
    value: float,
) -> str:
    return (
        "N/A"
        if math.isnan(value)
        else f"{value:.6f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    args = parse_args()

    input_path = Path(
        args.input
    )

    results_output_path = Path(
        args.results_output
    )

    summary_output_path = Path(
        args.summary_output
    )

    dataset_rows = load_eval_rows(
        input_path
    )

    if args.limit > 0:
        dataset_rows = dataset_rows[
            : args.limit
        ]

    LOGGER.info(
        "Loaded %d evaluation questions from %s",
        len(dataset_rows),
        input_path,
    )

    results: list[
        dict[str, str]
    ] = []

    total = len(
        dataset_rows
    )

    for index, row in enumerate(
        dataset_rows,
        start=1,
    ):
        LOGGER.info(
            "Answering sample %d/%d",
            index,
            total,
        )

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
            result = run_ragas(
                records,
                args.model,
            )

            print("=" * 80)
            print("RAGAS RESULT")
            print(result)
            print("=" * 80)

            score_rows = (
                extract_score_rows(
                    result
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

            if len(score_rows) != len(
                answered_indexes
            ):
                LOGGER.warning(
                    "RAGAS returned %d score rows "
                    "for %d answered rows.",
                    len(score_rows),
                    len(answered_indexes),
                )

        except Exception:
            LOGGER.exception(
                "RAGAS evaluation failed."
            )

    else:
        LOGGER.warning(
            "No answered rows available for RAGAS."
        )

    averages = (
        compute_metric_averages(
            results
        )
    )

    save_results(
        results_output_path,
        results,
    )

    save_summary(
        summary_output_path,
        averages,
    )

    counts = summarize_counts(
        results
    )

    LOGGER.info(
        "Saved evaluation results to %s",
        results_output_path,
    )

    LOGGER.info(
        "Saved RAGAS summary to %s",
        summary_output_path,
    )

    print(
        f"Total questions: {counts['total']}"
    )

    print(
        f"Answered: {counts['answered']}"
    )

    print(
        f"Blocked: {counts['blocked']}"
    )

    print(
        f"Failed: {counts['failed']}"
    )

    print(
        "Average Faithfulness: "
        f"{format_summary_value(averages['faithfulness'])}"
    )

    print(
        "Average Answer Relevancy: "
        f"{format_summary_value(averages['answer_relevancy'])}"
    )

    print(
        "Average Context Precision: "
        f"{format_summary_value(averages['context_precision'])}"
    )

    print(
        "Average Context Recall: "
        f"{format_summary_value(averages['context_recall'])}"
    )


if __name__ == "__main__":
    main()