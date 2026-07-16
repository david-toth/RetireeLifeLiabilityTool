from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class PremiumModel(Protocol):
    name: str
    is_expected_cashflow: bool

    def annual_premium(
        self,
        participant: pd.Series,
        attained_age: float,
        benefit_amount: float,
        death_benefit_cashflow: float,
        survival_start: float,
        duration: int,
    ) -> float:
        ...


@dataclass(frozen=True)
class ParticipantPremiumModel:
    name: str = "participant_annual_premium"
    is_expected_cashflow: bool = False

    def annual_premium(
        self,
        participant: pd.Series,
        attained_age: float,
        benefit_amount: float,
        death_benefit_cashflow: float,
        survival_start: float,
        duration: int,
    ) -> float:
        return float(participant["annual_premium"])


@dataclass(frozen=True)
class FlatAnnualPremiumModel:
    amount: float
    name: str = "flat_annual_premium"
    is_expected_cashflow: bool = False

    def annual_premium(
        self,
        participant: pd.Series,
        attained_age: float,
        benefit_amount: float,
        death_benefit_cashflow: float,
        survival_start: float,
        duration: int,
    ) -> float:
        return float(self.amount)


@dataclass(frozen=True)
class FlatRatePerThousandModel:
    rate_per_1000: float
    name: str = "flat_rate_per_1000"
    is_expected_cashflow: bool = False

    def annual_premium(
        self,
        participant: pd.Series,
        attained_age: float,
        benefit_amount: float,
        death_benefit_cashflow: float,
        survival_start: float,
        duration: int,
    ) -> float:
        return benefit_amount / 1000.0 * float(self.rate_per_1000)


@dataclass(frozen=True)
class AgeRatePerThousandModel:
    rates: pd.DataFrame
    name: str = "age_rate_per_1000"
    is_expected_cashflow: bool = False

    def __post_init__(self) -> None:
        required = {"age", "rate_per_1000"}
        missing = required.difference(self.rates.columns)
        if missing:
            raise ValueError(f"Premium rate table is missing required columns: {sorted(missing)}")
        clean = self.rates.loc[:, ["age", "rate_per_1000"]].copy()
        clean["age"] = pd.to_numeric(clean["age"], errors="coerce")
        clean["rate_per_1000"] = pd.to_numeric(clean["rate_per_1000"], errors="coerce")
        clean = clean.dropna().sort_values("age")
        if clean.empty:
            raise ValueError("Premium rate table must contain at least one usable age/rate_per_1000 row.")
        object.__setattr__(self, "rates", clean)
        object.__setattr__(self, "_ages", clean["age"].to_numpy(float))
        object.__setattr__(self, "_rates", clean["rate_per_1000"].to_numpy(float))

    def annual_premium(
        self,
        participant: pd.Series,
        attained_age: float,
        benefit_amount: float,
        death_benefit_cashflow: float,
        survival_start: float,
        duration: int,
    ) -> float:
        ages = self._ages
        rates = self._rates
        rate = float(np.interp(attained_age, ages, rates, left=rates[0], right=rates[-1]))
        return benefit_amount / 1000.0 * rate


@dataclass(frozen=True)
class TargetLossRatioPremiumModel:
    target_loss_ratio: float
    name: str = "target_loss_ratio"
    is_expected_cashflow: bool = True

    def __post_init__(self) -> None:
        if self.target_loss_ratio <= 0:
            raise ValueError("Target loss ratio must be greater than zero.")

    def annual_premium(
        self,
        participant: pd.Series,
        attained_age: float,
        benefit_amount: float,
        death_benefit_cashflow: float,
        survival_start: float,
        duration: int,
    ) -> float:
        return death_benefit_cashflow / float(self.target_loss_ratio)


@dataclass(frozen=True)
class CurrentPremiumToTargetLossRatioModel:
    target_loss_ratio: float
    annual_trend: float = 0.0
    grade_years: int = 5
    name: str = "current_premium_to_target_loss_ratio"
    is_expected_cashflow: bool = True

    def __post_init__(self) -> None:
        if self.target_loss_ratio <= 0:
            raise ValueError("Target loss ratio must be greater than zero.")
        if self.grade_years < 0:
            raise ValueError("Grade years cannot be negative.")

    def annual_premium(
        self,
        participant: pd.Series,
        attained_age: float,
        benefit_amount: float,
        death_benefit_cashflow: float,
        survival_start: float,
        duration: int,
    ) -> float:
        current_projection = (
            float(survival_start)
            * float(participant["annual_premium"])
            * ((1.0 + float(self.annual_trend)) ** int(duration))
        )
        target_projection = death_benefit_cashflow / float(self.target_loss_ratio)
        if self.grade_years == 0:
            current_weight = 0.0
        else:
            current_weight = max(0.0, (float(self.grade_years) - int(duration)) / float(self.grade_years))
        target_weight = 1.0 - current_weight
        return current_weight * current_projection + target_weight * target_projection
