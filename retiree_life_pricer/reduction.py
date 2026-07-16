from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


ReductionBasis = Literal["age", "duration", "calendar_year"]


@dataclass
class ReductionSchedules:
    schedules: pd.DataFrame
    rules: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        schedules = self._normalise_columns(self.schedules)
        rules = self.rules
        if rules is None and "type" in schedules.columns:
            rule_mask = schedules["type"].notna() & (schedules["type"].astype(str).str.strip() != "")
            rules = schedules.loc[rule_mask].copy()
            schedules = schedules.loc[~rule_mask].copy()
        self.schedules = self._normalise_table(schedules)
        if rules is not None:
            self.rules = self._normalise_rules(rules)
        self._tables_by_id = {
            str(schedule_id): group
            for schedule_id, group in self.schedules.groupby("schedule_id", sort=False)
        }
        self._rules_by_id = {}
        if self.rules is not None and not self.rules.empty:
            self._rules_by_id = {
                str(schedule_id): group
                for schedule_id, group in self.rules.groupby("schedule_id", sort=False)
            }
        self._known_schedule_ids = set(self._tables_by_id) | set(self._rules_by_id)

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
        return out

    @staticmethod
    def _normalise_table(schedules: pd.DataFrame) -> pd.DataFrame:
        out = ReductionSchedules._normalise_columns(schedules)
        if out.empty:
            return pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"])
        required = {"schedule_id", "basis", "point", "factor"}
        missing = required.difference(out.columns)
        if missing and "type" in out.columns:
            return pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"])
        if missing:
            raise ValueError(f"Reduction schedule is missing columns: {sorted(missing)}")
        out["schedule_id"] = out["schedule_id"].astype(str)
        out["basis"] = out["basis"].astype(str).str.lower()
        out["point"] = pd.to_numeric(out["point"], errors="coerce")
        out["factor"] = pd.to_numeric(out["factor"], errors="coerce").clip(0.0)
        return out.dropna(subset=["point", "factor"]).sort_values(["schedule_id", "basis", "point"])

    @staticmethod
    def _normalise_rules(rules: pd.DataFrame) -> pd.DataFrame:
        out = ReductionSchedules._normalise_columns(rules)
        required = {"schedule_id", "type"}
        missing = required.difference(out.columns)
        if missing:
            raise ValueError(f"Reduction rules are missing columns: {sorted(missing)}")
        out["schedule_id"] = out["schedule_id"].astype(str)
        out["type"] = out["type"].astype(str).str.lower()
        numeric_defaults = {
            "start_age": 0.0,
            "monthly_reduction": 0.0,
            "annual_reduction": 0.0,
            "reduction": 0.0,
            "minimum_factor": 0.0,
            "minimum_amount": 0.0,
            "age": np.nan,
            "amount": np.nan,
            "factor": np.nan,
        }
        for column, default in numeric_defaults.items():
            if column not in out.columns:
                out[column] = default
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
        if "period" not in out.columns:
            out["period"] = ""
        out["period"] = out["period"].astype(str).str.lower()
        return out

    @classmethod
    def from_file(cls, path: str | Path) -> "ReductionSchedules":
        path = Path(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        return cls(df)

    @classmethod
    def from_files(cls, table_path: str | Path | None = None, rules_path: str | Path | None = None) -> "ReductionSchedules":
        tables = pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"])
        rules = None
        if table_path is not None:
            tables = cls.from_file(table_path).schedules
        if rules_path is not None:
            path = Path(rules_path)
            if path.suffix.lower() in {".xlsx", ".xls"}:
                rules = pd.read_excel(path)
            else:
                rules = pd.read_csv(path)
        return cls(tables, rules=rules)

    def _rule_factor(self, schedule_id: str, attained_age: float, coverage_amount: float | None) -> float | None:
        if self.rules is None or self.rules.empty:
            return None
        subset = self._rules_by_id.get(str(schedule_id))
        if subset is None or subset.empty:
            return None
        rule = subset.iloc[0]
        if rule["type"] == "monthly_stepdown":
            return percent_stepdown_factor(
                attained_age=attained_age,
                start_age=float(rule["start_age"]),
                reduction=float(rule["monthly_reduction"]),
                minimum_factor=float(rule["minimum_factor"]),
                periods_per_year=12,
            )
        if rule["type"] == "annual_stepdown":
            return percent_stepdown_factor(
                attained_age=attained_age,
                start_age=float(rule["start_age"]),
                reduction=float(rule["annual_reduction"]),
                minimum_factor=float(rule["minimum_factor"]),
                periods_per_year=1,
            )
        if rule["type"] == "percent_stepdown":
            periods_per_year = 12 if rule["period"] == "monthly" else 1
            reduction = float(rule["reduction"])
            return percent_stepdown_factor(
                attained_age=attained_age,
                start_age=float(rule["start_age"]),
                reduction=reduction,
                minimum_factor=float(rule["minimum_factor"]),
                periods_per_year=periods_per_year,
            )
        if rule["type"] == "fixed_amount_by_age":
            if coverage_amount is None or coverage_amount <= 0:
                raise ValueError("fixed_amount_by_age rules require positive participant coverage_amount.")
            return fixed_amount_by_age_factor(
                rules=subset,
                attained_age=attained_age,
                coverage_amount=coverage_amount,
            )
        if rule["type"] == "fixed_percent_by_age":
            return fixed_percent_by_age_factor(rules=subset, attained_age=attained_age)
        raise ValueError(f"Unsupported reduction rule type: {rule['type']}")

    def has_schedule(self, schedule_id: str | None) -> bool:
        if schedule_id is None or pd.isna(schedule_id):
            return False
        return str(schedule_id) in self._known_schedule_ids

    def factor(
        self,
        schedule_id: str,
        attained_age: float,
        duration: float,
        calendar_year: int,
        default_schedule_id: str = "default",
        coverage_amount: float | None = None,
    ) -> float:
        schedule_id = str(schedule_id) if pd.notna(schedule_id) else default_schedule_id
        rule_factor = self._rule_factor(schedule_id, attained_age, coverage_amount)
        if rule_factor is not None:
            return rule_factor
        subset = self._tables_by_id.get(schedule_id)
        if subset is None or subset.empty:
            subset = self._tables_by_id.get(str(default_schedule_id))
        if subset is None or subset.empty:
            return 1.0
        basis = subset["basis"].iloc[0]
        value = {"age": attained_age, "duration": duration, "calendar_year": calendar_year}.get(basis, attained_age)
        return float(np.interp(value, subset["point"].to_numpy(float), subset["factor"].to_numpy(float)))


def percent_stepdown_factor(
    attained_age: float,
    start_age: float,
    reduction: float,
    minimum_factor: float,
    periods_per_year: int,
) -> float:
    if attained_age < start_age:
        return 1.0
    periods_elapsed = int(np.floor((attained_age - start_age) * periods_per_year + 1e-9))
    return float(max(1.0 - reduction * periods_elapsed, minimum_factor))


def monthly_stepdown_factor(
    attained_age: float,
    start_age: float,
    monthly_reduction: float,
    minimum_factor: float,
) -> float:
    return percent_stepdown_factor(attained_age, start_age, monthly_reduction, minimum_factor, 12)


def annual_stepdown_factor(
    attained_age: float,
    start_age: float,
    annual_reduction: float,
    minimum_factor: float,
) -> float:
    return percent_stepdown_factor(attained_age, start_age, annual_reduction, minimum_factor, 1)


def fixed_amount_by_age_factor(
    rules: pd.DataFrame,
    attained_age: float,
    coverage_amount: float,
) -> float:
    clean = rules.dropna(subset=["age", "amount"]).sort_values("age")
    if clean.empty:
        return 1.0
    applicable = clean[clean["age"] <= attained_age]
    if applicable.empty:
        return 1.0
    amount = float(applicable.iloc[-1]["amount"])
    return float(max(amount / coverage_amount, 0.0))


def fixed_percent_by_age_factor(rules: pd.DataFrame, attained_age: float) -> float:
    clean = rules.dropna(subset=["age", "factor"]).sort_values("age")
    if clean.empty:
        return 1.0
    applicable = clean[clean["age"] <= attained_age]
    if applicable.empty:
        return 1.0
    return float(max(float(applicable.iloc[-1]["factor"]), 0.0))


def monthly_stepdown_rule(
    schedule_id: str,
    start_age: float = 65.0,
    monthly_reduction: float = 0.025,
    minimum_factor: float = 1.0 / 3.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "schedule_id": schedule_id,
                "type": "monthly_stepdown",
                "start_age": start_age,
                "monthly_reduction": monthly_reduction,
                "minimum_factor": minimum_factor,
            }
        ]
    )


def annual_stepdown_rule(
    schedule_id: str,
    start_age: float = 65.0,
    annual_reduction: float = 0.10,
    minimum_factor: float = 0.50,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "schedule_id": schedule_id,
                "type": "annual_stepdown",
                "start_age": start_age,
                "annual_reduction": annual_reduction,
                "minimum_factor": minimum_factor,
            }
        ]
    )


def fixed_amount_by_age_rule(schedule_id: str, age_amounts: pd.DataFrame) -> pd.DataFrame:
    out = age_amounts.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "age" not in out.columns or "amount" not in out.columns:
        raise ValueError("Fixed amount by age rules require age and amount columns.")
    out["schedule_id"] = schedule_id
    out["type"] = "fixed_amount_by_age"
    return out.loc[:, ["schedule_id", "type", "age", "amount"]]


def fixed_percent_by_age_rule(schedule_id: str, age_factors: pd.DataFrame) -> pd.DataFrame:
    out = age_factors.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "age" not in out.columns or "factor" not in out.columns:
        raise ValueError("Fixed percent by age rules require age and factor columns.")
    out["schedule_id"] = schedule_id
    out["type"] = "fixed_percent_by_age"
    return out.loc[:, ["schedule_id", "type", "age", "factor"]]


def preview_monthly_stepdown(
    start_age: float = 65.0,
    monthly_reduction: float = 0.025,
    minimum_factor: float = 1.0 / 3.0,
    months: int = 60,
) -> pd.DataFrame:
    rows = []
    for month in range(months + 1):
        age = start_age + month / 12.0
        rows.append(
            {
                "month": month,
                "attained_age": age,
                "benefit_factor": monthly_stepdown_factor(age, start_age, monthly_reduction, minimum_factor),
            }
        )
    return pd.DataFrame(rows)


def preview_rule(
    rules: pd.DataFrame,
    coverage_amount: float,
    start_age: float,
    years: int,
) -> pd.DataFrame:
    schedule_id = str(rules.iloc[0]["schedule_id"])
    reductions = ReductionSchedules(pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]), rules=rules)
    rows = []
    for year in range(years + 1):
        attained_age = start_age + year
        factor = reductions.factor(
            schedule_id=schedule_id,
            attained_age=attained_age,
            duration=year,
            calendar_year=0,
            coverage_amount=coverage_amount,
        )
        rows.append(
            {
                "projection_year": year + 1,
                "attained_age": attained_age,
                "benefit_factor": factor,
                "coverage_amount": coverage_amount * factor,
            }
        )
    return pd.DataFrame(rows)


def level_schedule() -> ReductionSchedules:
    return ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}]))
