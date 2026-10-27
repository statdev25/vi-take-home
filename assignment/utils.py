"""Shared helpers."""

import pandas as pd

TARGET_COLUMNS = ["patient_token", "score", "program_recommended"]


def validate_targets(targets: pd.DataFrame) -> None:
    """Raise `ValueError` if `targets` is not a well-formed ranked outreach list.

    Extra columns are allowed; the required ones must be present and clean.
    """
    missing = [c for c in TARGET_COLUMNS if c not in targets.columns]
    if missing:
        raise ValueError(f"targets is missing columns: {missing}")
    if targets.empty:
        raise ValueError("targets is empty")
    if targets["patient_token"].isna().any():
        raise ValueError("patient_token must not be null")
    if targets["patient_token"].duplicated().any():
        raise ValueError("patient_token must be unique; a patient appears twice")
    if not pd.api.types.is_numeric_dtype(targets["score"]):
        raise ValueError("score must be numeric")
    if targets["score"].isna().any():
        raise ValueError("score must not be null")
    if targets["program_recommended"].isna().any():
        raise ValueError("program_recommended must not be null")
