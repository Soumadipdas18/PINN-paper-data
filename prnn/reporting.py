"""Paper-facing tables and concise method settings."""
from __future__ import annotations
import re
import pandas as pd

# Results and metrics are retained. These labels duplicate other columns.
TEXT_COLUMNS_TO_OMIT = {
    "Stages A-D runs": ("run_name",),
    "Stage E runs": ("run_name", "comparison_type"),
    "Main training history": ("run_name",),
}

SETUP_PARAMETERS_TO_OMIT = {
    "Dataset SHA-256", "Cooling trajectory alignment", "Retention spatial domain",
    "Resolution quadrature diagnostic", "Near-target empty subset", "Design margin basis",
    "Inference timing", "Optimization status", "Verification status", "Selection limitation",
    "Worksheet count", "Nearest-neighbor distance",
}
SETUP_SECTIONS_TO_OMIT = {"Provenance", "Workbook", "Execution", "Holdout hardware"}


def paper_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare an independent display table without changing numeric results."""
    result = frame.drop(columns=list(TEXT_COLUMNS_TO_OMIT.get(name, ())), errors="ignore").copy()
    if name == "Run setup":
        section = result["Section"].astype(str)
        parameter = result["Parameter"].astype(str)
        keep = ~section.isin(SETUP_SECTIONS_TO_OMIT)
        keep &= ~parameter.isin(SETUP_PARAMETERS_TO_OMIT)
        keep &= ~parameter.str.contains(r"sha[-_ ]?256|checksum|fingerprint|cache", case=False, regex=True)
        result = result.loc[keep].copy()
        selected = result["Parameter"].eq("Resolution selection rule")
        result.loc[selected, "Value"] = "Smallest resolution within 1% of the lowest validation score."
        selected = result["Parameter"].str.lower().eq("python")
        result.loc[selected, "Value"] = result.loc[selected, "Value"].map(lambda value: str(value).split()[0])
        result["Units or definition"] = result["Units or definition"].replace({
            "cases/hour, supplied assumption": "cases/hour (assumed)",
            "hours, estimate not measured wall-clock time": "hours (sequential estimate)",
        })
    if name == "Holdout protocols":
        result["definition"] = result["definition"].replace({
            "Original deterministic stratified 70/15/15 split": "Stratified 70/15/15 split",
        })
    return result.reset_index(drop=True)


def paper_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: paper_table(name, frame) for name, frame in tables.items()}
