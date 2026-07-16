from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .models import PricingAssumptions, validate_participants
from .mortality import ImprovementScale, MortalityTable
from .premium import ParticipantPremiumModel, PremiumModel
from .reduction import ReductionSchedules
from .yield_curve import YieldCurve


@dataclass
class PricingEngine:
    mortality: MortalityTable
    yield_curve: YieldCurve
    reductions: ReductionSchedules
    improvement: ImprovementScale | None = None
    cohort_mortality: dict[str, MortalityTable] | None = None
    premium_model: PremiumModel | None = None

    @staticmethod
    def _cache_age(age: float) -> float:
        return round(float(age), 8)

    def _mortality_table(self, cohort: str) -> MortalityTable:
        if self.cohort_mortality is None:
            return self.mortality
        return self.cohort_mortality.get(cohort, self.mortality)

    def _mortality_rate(
        self,
        sex: str,
        age: float,
        cohort: str,
        cache: dict[tuple[int, str, float], tuple[float, str]] | None = None,
    ) -> tuple[float, str]:
        table = self._mortality_table(cohort)
        key = (id(table), str(sex).strip().upper(), self._cache_age(age))
        if cache is not None and key in cache:
            return cache[key]
        value = (table.qx(sex, age), table.name)
        if cache is not None:
            cache[key] = value
        return value

    def _improvement_factor(
        self,
        sex: str,
        age: float,
        duration: int,
        assumptions: PricingAssumptions,
        cache: dict[tuple[Any, ...], float] | None = None,
    ) -> float:
        if not assumptions.mortality_improvement or self.improvement is None:
            return 1.0
        key = (
            id(self.improvement),
            str(sex).strip().upper(),
            self._cache_age(age),
            int(duration),
            int(assumptions.valuation_date.year),
            int(assumptions.mortality_base_year),
        )
        if cache is not None and key in cache:
            return cache[key]
        value = self.improvement.factor(
            sex=sex,
            age=age,
            years_from_valuation=duration,
            valuation_year=assumptions.valuation_date.year,
            base_year=assumptions.mortality_base_year,
        )
        if cache is not None:
            cache[key] = value
        return value

    def _reduction_schedule_id(self, participant: pd.Series, assumptions: PricingAssumptions) -> str:
        default_schedule_id = str(assumptions.default_reduction_schedule_id)
        if self.reductions.has_schedule(default_schedule_id):
            return default_schedule_id
        participant_schedule_id = participant["reduction_schedule_id"]
        if pd.notna(participant_schedule_id) and self.reductions.has_schedule(str(participant_schedule_id)):
            return str(participant_schedule_id)
        return ""

    def project(self, participants: pd.DataFrame, assumptions: PricingAssumptions) -> tuple[pd.DataFrame, pd.DataFrame]:
        participants = validate_participants(participants, assumptions.valuation_date)
        rows = []
        premium_model = self.premium_model or ParticipantPremiumModel()
        mortality_cache: dict[tuple[int, str, float], tuple[float, str]] = {}
        improvement_cache: dict[tuple[Any, ...], float] = {}
        reduction_cache: dict[tuple[str, float, float, int, float], float] = {}
        benefit_offsets = {"mid_year": 0.5, "end_of_year": 1.0}
        premium_offsets = {
            "beginning_of_year": 0.0,
            "mid_year": 0.5,
            "end_of_year": 1.0,
        }
        benefit_offset = benefit_offsets[assumptions.benefit_timing]
        premium_offset = premium_offsets[assumptions.premium_timing]
        benefit_dfs = [
            self.yield_curve.discount_factor(duration + benefit_offset)
            for duration in range(assumptions.projection_years)
        ]
        premium_dfs = [
            self.yield_curve.discount_factor(duration + premium_offset)
            for duration in range(assumptions.projection_years)
        ]
        participant_records = participants.to_dict("records")

        for p in participant_records:
            survival = 1.0
            premium_end_age = (
                float(p["premium_end_age"])
                if pd.notna(p["premium_end_age"])
                else assumptions.default_premium_end_age
            )
            schedule_id = self._reduction_schedule_id(p, assumptions)
            coverage_start_age = float(p["coverage_start_age"]) if pd.notna(p["coverage_start_age"]) else float(p["age"])
            initial_coverage = float(p["coverage_amount"])
            mortality_multiplier = float(p["mortality_multiplier"])

            for duration in range(assumptions.projection_years):
                attained_age = float(p["age"]) + duration
                calendar_year = assumptions.valuation_date.year + duration
                sex = str(p["sex"])
                cohort = str(p["cohort"]) if pd.notna(p["cohort"]) else assumptions.default_cohort
                base_qx, mortality_table = self._mortality_rate(sex, attained_age, cohort, mortality_cache)
                improvement_factor = self._improvement_factor(
                    sex,
                    attained_age,
                    duration,
                    assumptions,
                    improvement_cache,
                )
                cohort_multiplier = assumptions.cohort_mortality_multipliers.get(cohort, 1.0)
                qx = min(1.0, base_qx * improvement_factor * mortality_multiplier * cohort_multiplier)

                reduction_duration = max(0.0, attained_age - coverage_start_age)
                reduction_key = (
                    schedule_id,
                    self._cache_age(attained_age),
                    self._cache_age(reduction_duration),
                    int(calendar_year),
                    initial_coverage,
                )
                if reduction_key in reduction_cache:
                    benefit_factor = reduction_cache[reduction_key]
                else:
                    benefit_factor = self.reductions.factor(
                        schedule_id=schedule_id,
                        attained_age=attained_age,
                        duration=reduction_duration,
                        calendar_year=calendar_year,
                        default_schedule_id=assumptions.default_reduction_schedule_id,
                        coverage_amount=initial_coverage,
                    )
                    reduction_cache[reduction_key] = benefit_factor
                benefit_amount = initial_coverage * benefit_factor
                expected_deaths = survival * qx
                death_benefit = expected_deaths * benefit_amount

                annual_premium = premium_model.annual_premium(
                    participant=p,
                    attained_age=attained_age,
                    benefit_amount=benefit_amount,
                    death_benefit_cashflow=death_benefit,
                    survival_start=survival,
                    duration=duration,
                )
                premium_payable = 0.0
                if attained_age < premium_end_age:
                    if premium_model.is_expected_cashflow:
                        premium_payable = annual_premium
                    else:
                        premium_payable = survival * annual_premium

                benefit_df = benefit_dfs[duration]
                premium_df = premium_dfs[duration]
                rows.append(
                    {
                        "participant_id": p["participant_id"],
                        "sex": sex,
                        "date_of_birth": p["date_of_birth"],
                        "cohort": cohort,
                        "mortality_table": mortality_table,
                        "coverage_amount": initial_coverage,
                        "annual_premium": annual_premium,
                        "premium_basis": premium_model.name,
                        "premium_end_age": premium_end_age,
                        "reduction_schedule_id": schedule_id,
                        "valuation_age": float(p["age"]),
                        "projection_year": duration + 1,
                        "calendar_year": calendar_year,
                        "attained_age": attained_age,
                        "survival_start": survival,
                        "qx": qx,
                        "improvement_factor": improvement_factor,
                        "benefit_factor": benefit_factor,
                        "benefit_amount": benefit_amount,
                        "expected_deaths": expected_deaths,
                        "death_benefit_cashflow": death_benefit,
                        "premium_cashflow": premium_payable,
                        "pv_death_benefit": death_benefit * benefit_df,
                        "pv_future_premium": premium_payable * premium_df,
                        "net_pv_liability": death_benefit * benefit_df - premium_payable * premium_df,
                    }
                )
                survival *= 1.0 - qx
                if survival <= 1e-10:
                    break

        cashflows = pd.DataFrame(rows)
        summary = self._summarise(cashflows)
        return cashflows, summary

    @staticmethod
    def _summarise(cashflows: pd.DataFrame) -> pd.DataFrame:
        if cashflows.empty:
            return pd.DataFrame()
        participant_fields = [
            "sex",
            "date_of_birth",
            "valuation_age",
            "cohort",
            "mortality_table",
            "coverage_amount",
            "annual_premium",
            "premium_basis",
            "premium_end_age",
            "reduction_schedule_id",
        ]
        return (
            cashflows.groupby("participant_id", as_index=False)
            .agg(
                **{field: (field, "first") for field in participant_fields},
                pv_death_benefit=("pv_death_benefit", "sum"),
                pv_future_premium=("pv_future_premium", "sum"),
                net_pv_liability=("net_pv_liability", "sum"),
                expected_deaths=("expected_deaths", "sum"),
                inforce_coverage=("benefit_amount", "first"),
            )
            .sort_values("participant_id")
        )

    @staticmethod
    def annual_cohort_summary(cashflows: pd.DataFrame) -> pd.DataFrame:
        if cashflows.empty:
            return pd.DataFrame()

        value_columns = [
            "death_benefit_cashflow",
            "premium_cashflow",
            "pv_death_benefit",
            "pv_future_premium",
            "net_pv_liability",
            "expected_deaths",
        ]
        by_cohort = cashflows.groupby(["projection_year", "calendar_year", "cohort"], as_index=False)[
            value_columns
        ].sum()
        total = cashflows.groupby(["projection_year", "calendar_year"], as_index=False)[value_columns].sum()
        total["cohort"] = "Total"
        out = pd.concat([by_cohort, total], ignore_index=True, sort=False)
        out["total_cashflow"] = out["death_benefit_cashflow"] - out["premium_cashflow"]
        columns = ["projection_year", "calendar_year", "cohort", *value_columns, "total_cashflow"]
        return out.loc[:, columns].sort_values(["projection_year", "cohort"]).reset_index(drop=True)
