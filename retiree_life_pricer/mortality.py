from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    aliases = {
        "gender": "sex",
        "qx_rate": "qx",
        "mortality_rate": "qx",
        "rate": "qx",
        "attained_age": "age",
        "calendar_year": "year",
        "mi": "improvement",
        "improvement_rate": "improvement",
    }
    return out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})


@dataclass
class MortalityTable:
    rates: pd.DataFrame
    name: str = "custom"

    def __post_init__(self) -> None:
        rates = _normalise_columns(self.rates)
        required = {"sex", "age", "qx"}
        missing = required.difference(rates.columns)
        if missing:
            raise ValueError(f"Mortality table is missing columns: {sorted(missing)}")
        rates = rates.loc[:, ["sex", "age", "qx"]].copy()
        rates["sex"] = rates["sex"].astype(str).str.strip().str.upper()
        rates["age"] = pd.to_numeric(rates["age"], errors="coerce")
        rates["qx"] = pd.to_numeric(rates["qx"], errors="coerce").clip(0.0, 1.0)
        rates = rates.dropna(subset=["sex", "age", "qx"]).sort_values(["sex", "age"])
        self.rates = rates
        self._rates_by_sex = {
            sex: (
                group["age"].to_numpy(dtype=float),
                group["qx"].to_numpy(dtype=float),
            )
            for sex, group in rates.groupby("sex", sort=False)
        }
        self._all_ages = rates["age"].to_numpy(dtype=float)
        self._all_qxs = rates["qx"].to_numpy(dtype=float)

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> "MortalityTable":
        path = Path(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        return cls(df, name=name or path.stem)

    @classmethod
    def from_url(cls, url: str, name: str = "url_mortality") -> "MortalityTable":
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        from io import StringIO

        return cls(pd.read_csv(StringIO(response.text)), name=name)

    def qx(self, sex: str, age: float) -> float:
        """Return the one-year death probability from age to age + 1 under UDD."""
        sex = str(sex).strip().upper()
        ages, qxs = self._rates_by_sex.get(sex, (self._all_ages, self._all_qxs))
        if len(ages) == 0:
            raise ValueError("Mortality table has no usable rates.")
        return _one_year_udd_qx(age, ages, qxs)


def _table_qx(age: float, ages: np.ndarray, qxs: np.ndarray) -> float:
    return float(np.interp(age, ages, qxs, left=qxs[0], right=1.0))


def _one_year_udd_qx(age: float, ages: np.ndarray, qxs: np.ndarray) -> float:
    if age >= float(np.max(ages)):
        return 1.0

    floor_age = np.floor(age)
    fraction = float(age - floor_age)
    q_current = _table_qx(floor_age, ages, qxs)

    if fraction == 0.0:
        return q_current

    q_next = _table_qx(floor_age + 1.0, ages, qxs)
    survival_to_next_integer = (1.0 - q_current) / max(1e-12, 1.0 - fraction * q_current)
    survival_after_next_integer = 1.0 - fraction * q_next
    return float(np.clip(1.0 - survival_to_next_integer * survival_after_next_integer, 0.0, 1.0))


@dataclass
class ImprovementScale:
    rates: pd.DataFrame
    name: str = "custom_improvement"

    def __post_init__(self) -> None:
        rates = _normalise_columns(self.rates)
        required = {"sex", "age", "year", "improvement"}
        missing = required.difference(rates.columns)
        if missing:
            raise ValueError(f"Improvement scale is missing columns: {sorted(missing)}")
        rates = rates.loc[:, ["sex", "age", "year", "improvement"]].copy()
        rates["sex"] = rates["sex"].astype(str).str.strip().str.upper()
        for column in ["age", "year", "improvement"]:
            rates[column] = pd.to_numeric(rates[column], errors="coerce")
        self.rates = rates.dropna().sort_values(["sex", "year", "age"])
        self._rates_by_sex_year = {
            (sex, int(year)): (
                group["age"].to_numpy(dtype=float),
                group["improvement"].to_numpy(dtype=float),
            )
            for (sex, year), group in self.rates.groupby(["sex", "year"], sort=False)
        }
        self._rates_by_year = {
            int(year): (
                group["age"].to_numpy(dtype=float),
                group["improvement"].to_numpy(dtype=float),
            )
            for year, group in self.rates.groupby("year", sort=False)
        }

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> "ImprovementScale":
        path = Path(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        return cls(df, name=name or path.stem)

    def factor(
        self,
        sex: str,
        age: float,
        years_from_valuation: int,
        valuation_year: int,
        base_year: int | None = None,
    ) -> float:
        start_year = int(base_year) if base_year is not None else int(valuation_year)
        target_year = int(valuation_year) + int(years_from_valuation)
        if target_year <= start_year:
            return 1.0
        factor = 1.0
        sex = str(sex).strip().upper()
        for year in range(start_year, target_year):
            arrays = self._rates_by_sex_year.get((sex, year))
            if arrays is None:
                arrays = self._rates_by_year.get(year)
            if arrays is None:
                continue
            ages, improvements = arrays
            improvement = float(np.interp(age, ages, improvements))
            factor *= max(0.0, 1.0 - improvement)
        return factor


def sample_mortality_table() -> MortalityTable:
    rows = []
    for sex, base, slope in [("M", 0.008, 1.095), ("F", 0.006, 1.088)]:
        for age in range(50, 121):
            qx = min(1.0, base * (slope ** (age - 65)))
            rows.append({"sex": sex, "age": age, "qx": qx})
    return MortalityTable(pd.DataFrame(rows), name="illustrative")
