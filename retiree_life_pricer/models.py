from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd


DiscountBasis = Literal["fixed", "spot_curve"]


@dataclass(frozen=True)
class PricingAssumptions:
    valuation_date: date
    projection_years: int = 80
    default_reduction_schedule_id: str = "default"
    default_premium_end_age: float = 120.0
    benefit_timing: Literal["end_of_year", "mid_year"] = "mid_year"
    premium_timing: Literal["beginning_of_year", "mid_year", "end_of_year"] = "beginning_of_year"
    mortality_improvement: bool = True
    mortality_base_year: int = 2012
    default_cohort: str = "default"
    cohort_mortality_multipliers: dict[str, float] = field(default_factory=dict)


PARTICIPANT_REQUIRED_COLUMNS = {"participant_id", "sex", "date_of_birth", "coverage_amount"}
PARTICIPANT_OPTIONAL_DEFAULTS = {
    "annual_premium": 0.0,
    "premium_end_age": pd.NA,
    "reduction_schedule_id": pd.NA,
    "mortality_multiplier": 1.0,
    "coverage_start_age": pd.NA,
    "cohort": pd.NA,
}


def _age_36525(date_of_birth: pd.Timestamp, valuation_date: pd.Timestamp) -> float:
    return float((valuation_date.normalize() - date_of_birth.normalize()).days / 365.25)


def _attained_age(date_of_birth: pd.Series, valuation_date: date) -> pd.Series:
    valuation_timestamp = pd.Timestamp(valuation_date)
    return date_of_birth.apply(lambda dob: _age_36525(dob, valuation_timestamp))


def validate_participants(df: pd.DataFrame, valuation_date: date | None = None) -> pd.DataFrame:
    missing = PARTICIPANT_REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Participant file is missing required columns: {sorted(missing)}")
    if valuation_date is None:
        raise ValueError("A valuation date is required to compute age from date_of_birth.")

    out = df.copy()
    for column, default in PARTICIPANT_OPTIONAL_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default

    out["participant_id"] = out["participant_id"].astype(str)
    out["sex"] = out["sex"].astype(str).str.strip().str.upper()
    out["cohort"] = out["cohort"].fillna("default").astype(str).str.strip().str.lower()

    out["date_of_birth"] = pd.to_datetime(out["date_of_birth"], errors="coerce")
    if out["date_of_birth"].isna().any():
        raise ValueError("date_of_birth must be a valid date for every participant.")
    out["age"] = _attained_age(out["date_of_birth"], valuation_date)

    numeric_columns = [
        "age",
        "coverage_amount",
        "annual_premium",
        "premium_end_age",
        "mortality_multiplier",
        "coverage_start_age",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out["age"].isna().any():
        raise ValueError("Participant age could not be computed from date_of_birth for every row.")
    if out["coverage_amount"].isna().any():
        raise ValueError("Coverage amount must be numeric for every row.")

    out["annual_premium"] = out["annual_premium"].fillna(0.0)
    out["mortality_multiplier"] = out["mortality_multiplier"].fillna(1.0)
    return out
