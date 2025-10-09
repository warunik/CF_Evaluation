"""Aggregate correctness, feasibility and timing metrics for counterfactuals."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from feasibility import load_and_analyze_feasibility
from Evaluation.cf_methods.method_registry import (
    DATASET_REGISTRY,
    METHOD_REGISTRY,
    DatasetConfig,
    MethodConfig,
)

LOGGER = logging.getLogger(__name__)

METRIC_ORDER = ("Correctness (%)", "Feasibility (%)", "Average Time (s)")


@dataclass
class EvaluationStats:
    """Aggregated evaluation statistics for a dataset/method pair."""

    correct_success: int = 0
    correct_total: int = 0
    feasible_success: int = 0
    feasible_total: int = 0
    time_sum: float = 0.0
    time_count: int = 0

    @property
    def correctness_pct(self) -> Optional[float]:
        if self.correct_total == 0:
            return None
        return (self.correct_success / self.correct_total) * 100.0

    @property
    def feasibility_pct(self) -> Optional[float]:
        if self.feasible_total == 0:
            return None
        return (self.feasible_success / self.feasible_total) * 100.0

    @property
    def average_time(self) -> Optional[float]:
        if self.time_count == 0:
            return None
        return self.time_sum / self.time_count

    def merge(self, other: "EvaluationStats") -> None:
        self.correct_success += other.correct_success
        self.correct_total += other.correct_total
        self.feasible_success += other.feasible_success
        self.feasible_total += other.feasible_total
        self.time_sum += other.time_sum
        self.time_count += other.time_count

    @classmethod
    def empty(cls) -> "EvaluationStats":
        return cls()


def _iter_dataset_files(
    dataset: DatasetConfig, method: MethodConfig
) -> Iterable[Tuple[Path, Path]]:
    """Yield available (report, validation) file pairs for a dataset/method."""

    if not method.results_dir.exists():
        LOGGER.debug("Results directory %s missing", method.results_dir)
        return []

    yielded: List[Tuple[Path, Path]] = []
    seen_reports: set[Path] = set()

    for prefix in dataset.prefixes:
        pattern = method.report_glob(prefix)
        for report_path in method.results_dir.glob(pattern):
            if report_path in seen_reports:
                continue

            validation_path = method.expected_validation_path(report_path)
            if not validation_path.exists():
                LOGGER.debug(
                    "Skipping report %s because %s is missing",
                    report_path.name,
                    validation_path,
                )
                continue

            yielded.append((report_path, validation_path))
            seen_reports.add(report_path)

    return yielded


def _find_time_column(columns: Sequence[str], hints: Sequence[str]) -> Optional[str]:
    for hint in hints:
        for column in columns:
            if hint.lower() in column.lower():
                return column
    return None


def _compute_correctness_counts(validation_path: Path) -> Tuple[int, int]:
    df = pd.read_csv(validation_path)
    if "result" not in df.columns:
        LOGGER.warning("Validation file %s lacks 'result' column", validation_path)
        return 0, 0

    mask = df["result"].isin(["✓", "✗"])
    total = int(mask.sum())
    success = int((df.loc[mask, "result"] == "✓").sum())
    return success, total


def _compute_time_stats(report_path: Path, hints: Sequence[str]) -> Tuple[float, int]:
    df = pd.read_csv(report_path)
    column = _find_time_column(df.columns, hints)
    if column is None:
        LOGGER.debug("Report %s lacks a time column", report_path)
        return 0.0, 0

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.sum()), int(values.count())


def _compute_feasibility_counts(
    dataset: DatasetConfig, report_path: Path, validation_path: Path
) -> Tuple[int, int]:
    if not dataset.original_data.exists():
        LOGGER.debug(
            "Original data for %s missing at %s", dataset.key, dataset.original_data
        )
        return 0, 0

    results = load_and_analyze_feasibility(
        str(report_path),
        str(validation_path),
        str(dataset.original_data),
        dataset.feasibility_key,
    )

    feasible = sum(1 for item in results if item.get("feasible"))
    total = len(results)
    return feasible, total


def evaluate_dataset_method(
    dataset: DatasetConfig, method: MethodConfig
) -> Optional[EvaluationStats]:
    stats = EvaluationStats.empty()
    any_data = False

    for report_path, validation_path in _iter_dataset_files(dataset, method):
        any_data = True

        correct_success, correct_total = _compute_correctness_counts(validation_path)
        stats.correct_success += correct_success
        stats.correct_total += correct_total

        feasible_success, feasible_total = _compute_feasibility_counts(
            dataset, report_path, validation_path
        )
        stats.feasible_success += feasible_success
        stats.feasible_total += feasible_total

        time_sum, time_count = _compute_time_stats(report_path, method.time_column_hints)
        stats.time_sum += time_sum
        stats.time_count += time_count

    if not any_data:
        return None

    return stats


def build_summary_dataframe(
    dataset_ids: Sequence[str], method_ids: Sequence[str]
) -> pd.DataFrame:
    data: Dict[Tuple[str, str], List[Optional[float]]] = {}
    index_labels: List[str] = []

    dataset_configs = [DATASET_REGISTRY[d_id] for d_id in dataset_ids]
    method_configs = [METHOD_REGISTRY[m_id] for m_id in method_ids]

    for method in method_configs:
        for metric in METRIC_ORDER:
            data[(method.display_name, metric)] = []

    for dataset in dataset_configs:
        index_labels.append(dataset.display_name)
        for method in method_configs:
            stats = evaluate_dataset_method(dataset, method)

            values = (
                stats.correctness_pct if stats else None,
                stats.feasibility_pct if stats else None,
                stats.average_time if stats else None,
            )

            for metric, value in zip(METRIC_ORDER, values):
                data[(method.display_name, metric)].append(value)

    columns = pd.MultiIndex.from_tuples(list(data.keys()))
    df = pd.DataFrame(data, index=index_labels, columns=columns)
    return df


def format_table_values(df: pd.DataFrame, float_format: str) -> pd.DataFrame:
    def _format(value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return "--"
        return format(value, float_format)

    formatted = df.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(_format)
    return formatted


def dataframe_to_markdown(
    df: pd.DataFrame, dataset_labels: Sequence[str], method_labels: Sequence[str]
) -> str:
    header_top = ["Dataset"]
    header_bottom = [""]

    for method in method_labels:
        header_top.extend([method, "", ""])
        header_bottom.extend(METRIC_ORDER)

    separator = ["---"] * len(header_top)

    lines = [
        "| " + " | ".join(header_top) + " |",
        "| " + " | ".join(separator) + " |",
        "| " + " | ".join(header_bottom) + " |",
    ]

    for label, (_, row) in zip(dataset_labels, df.iterrows()):
        row_values: List[str] = [label]
        for values in row.values.reshape(-1, len(METRIC_ORDER)):
            row_values.extend(values.tolist())
        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def dataframe_to_latex(df: pd.DataFrame, dataset_labels: Sequence[str]) -> str:
    method_count = len(df.columns) // len(METRIC_ORDER)
    column_spec = "l" + "ccc" * method_count
    latex_lines = ["\\begin{tabular}{" + column_spec + "}", "\\toprule"]

    method_headers = [col[0] for col in df.columns[:: len(METRIC_ORDER)]]
    header_line_1 = ["Dataset"]
    header_line_2 = [""]
    for method in method_headers:
        header_line_1.append(
            f"\\multicolumn{{{len(METRIC_ORDER)}}}{{c}}{{{method}}}"
        )
        header_line_2.extend(METRIC_ORDER)

    latex_lines.append(" & ".join(header_line_1) + "\\")
    latex_lines.append(" & ".join(header_line_2) + "\\")
    latex_lines.append("\\midrule")

    for label, (_, row) in zip(dataset_labels, df.iterrows()):
        values = " & ".join(row)
        latex_lines.append(f"{label} & {values}\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    return "\n".join(latex_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_REGISTRY.keys()),
        help="Dataset identifiers to include",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(METHOD_REGISTRY.keys()),
        help="Method identifiers to include",
    )
    parser.add_argument(
        "--float-format",
        default=".2f",
        help="Format specification used for numeric values (default: .2f)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save the raw numeric table as CSV",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional path to save the formatted table as Markdown",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        help="Optional path to save the formatted table as LaTeX",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    dataset_ids = [d for d in args.datasets if d in DATASET_REGISTRY]
    method_ids = [m for m in args.methods if m in METHOD_REGISTRY]

    if not dataset_ids:
        raise SystemExit("No valid datasets selected")
    if not method_ids:
        raise SystemExit("No valid methods selected")

    df_numeric = build_summary_dataframe(dataset_ids, method_ids)
    df_formatted = format_table_values(df_numeric, args.float_format)

    if args.output_csv:
        df_numeric.to_csv(args.output_csv)

    if args.output_markdown:
        args.output_markdown.write_text(
            dataframe_to_markdown(
                df_formatted, [DATASET_REGISTRY[d].display_name for d in dataset_ids],
                [METHOD_REGISTRY[m].display_name for m in method_ids],
            )
        )

    if args.output_latex:
        args.output_latex.write_text(
            dataframe_to_latex(
                df_formatted, [DATASET_REGISTRY[d].display_name for d in dataset_ids]
            )
        )

    print(df_formatted)


if __name__ == "__main__":
    main()
