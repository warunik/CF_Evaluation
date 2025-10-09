"""Registry of datasets and counterfactual methods for evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Metadata required to locate evaluation artefacts for a dataset.

    Attributes
    ----------
    key:
        Identifier used internally by the evaluation utilities.
    display_name:
        Human friendly name shown in summary tables.
    prefixes:
        Collection of filename prefixes associated with the dataset.  Counter-
        factual artefacts frequently follow the pattern
        ``{prefix}_{model}_{suffix}.csv`` and multiple prefixes are sometimes
        required (for instance ``german`` vs ``GermanCredit``).
    original_data:
        Path to the CSV file containing the original instances that were used
        to generate counterfactuals.  This file is required for feasibility
        analysis.
    feasibility_key:
        Dataset identifier expected by :class:`AutomatedFeasibilityAnalyzer` in
        :mod:`feasibility`.
    """

    key: str
    display_name: str
    prefixes: Tuple[str, ...]
    original_data: Path
    feasibility_key: str


@dataclass(frozen=True)
class MethodConfig:
    """Metadata describing how results for a counterfactual method are stored."""

    key: str
    display_name: str
    results_dir: Path
    validation_dir: Path = Path("validation_results")
    report_suffix: str = "_counterfactual_report.csv"
    validation_suffix: str = "_validation_summary.csv"
    time_column_hints: Tuple[str, ...] = ("time",)

    def report_glob(self, prefix: str) -> str:
        """Return the glob expression for reports of ``prefix``."""
        return f"{prefix}_*{self.report_suffix}"

    def expected_validation_path(self, report_path: Path) -> Path:
        """Return the validation summary path associated with ``report_path``."""
        return self.validation_dir / report_path.name.replace(
            self.report_suffix, self.validation_suffix
        )


def _dataset_path(filename: str) -> Path:
    return Path("data") / filename


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "adult": DatasetConfig(
        key="adult",
        display_name="Adult",
        prefixes=("adult",),
        original_data=_dataset_path("adult_test.csv"),
        feasibility_key="adult",
    ),
    "breastcancer": DatasetConfig(
        key="breastcancer",
        display_name="Breast Cancer",
        prefixes=("breastcancer", "breast_cancer"),
        original_data=_dataset_path("breastcancer_test.csv"),
        feasibility_key="breastcancer",
    ),
    "german": DatasetConfig(
        key="german",
        display_name="German Credit",
        prefixes=("german", "GermanCredit"),
        original_data=_dataset_path("GermanCredit_test.csv"),
        feasibility_key="german",
    ),
    "heart": DatasetConfig(
        key="heart",
        display_name="Heart Disease",
        prefixes=("heart", "HeartDisease"),
        original_data=_dataset_path("HeartDisease_test.csv"),
        feasibility_key="heart",
    ),
    "diabetes": DatasetConfig(
        key="diabetes",
        display_name="Diabetes",
        prefixes=("diabetes", "Diabetes"),
        original_data=_dataset_path("Diabetes_test.csv"),
        feasibility_key="diabetes",
    ),
    "iris": DatasetConfig(
        key="iris",
        display_name="Iris",
        prefixes=("iris",),
        original_data=_dataset_path("iris_test.csv"),
        feasibility_key="iris",
    ),
    "wine": DatasetConfig(
        key="wine",
        display_name="Wine Quality",
        prefixes=("wine",),
        original_data=_dataset_path("wine_test.csv"),
        feasibility_key="wine",
    ),
}


METHOD_REGISTRY: Dict[str, MethodConfig] = {
    "foil_tree": MethodConfig(
        key="foil_tree",
        display_name="Foil Tree",
        results_dir=Path("Evaluation/cf_methods/Foil_Trees/results"),
    ),
    "shap_c": MethodConfig(
        key="shap_c",
        display_name="SHAP-C",
        results_dir=Path("Evaluation/cf_methods/SHAP_C/results"),
    ),
    "lime_c": MethodConfig(
        key="lime_c",
        display_name="LIME-C",
        results_dir=Path("Evaluation/cf_methods/LIME_C/results"),
    ),
    "lore": MethodConfig(
        key="lore",
        display_name="LORE",
        results_dir=Path("Evaluation/cf_methods/LORE/results"),
    ),
    "clear": MethodConfig(
        key="clear",
        display_name="CLEAR",
        results_dir=Path("Evaluation/cf_methods/CLEAR/results"),
    ),
    "onb_macf": MethodConfig(
        key="onb_macf",
        display_name="ONB-MACF",
        results_dir=Path("Evaluation/cf_methods/ONB_MACF/results"),
    ),
    "mc3g": MethodConfig(
        key="mc3g",
        display_name="MC3G",
        results_dir=Path("Evaluation/cf_methods/MC3G/results"),
    ),
    "dicoflex": MethodConfig(
        key="dicoflex",
        display_name="DiCoFlex",
        results_dir=Path("Evaluation/cf_methods/DiCoFlex/results"),
    ),
}


def dataset_keys() -> Iterable[str]:
    """Return the dataset identifiers in insertion order."""

    return DATASET_REGISTRY.keys()


def method_keys() -> Iterable[str]:
    """Return the method identifiers in insertion order."""

    return METHOD_REGISTRY.keys()
