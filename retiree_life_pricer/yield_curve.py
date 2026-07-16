from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests


TREASURY_DAILY_CSV_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)

TREASURY_MATURITY_MAP = {
    "1 mo": 1 / 12,
    "2 mo": 2 / 12,
    "3 mo": 3 / 12,
    "4 mo": 4 / 12,
    "6 mo": 6 / 12,
    "1 yr": 1,
    "2 yr": 2,
    "3 yr": 3,
    "5 yr": 5,
    "7 yr": 7,
    "10 yr": 10,
    "20 yr": 20,
    "30 yr": 30,
}


@dataclass
class YieldCurve:
    terms: np.ndarray
    rates: np.ndarray
    name: str = "custom"

    def __post_init__(self) -> None:
        order = np.argsort(self.terms)
        self.terms = np.asarray(self.terms, dtype=float)[order]
        self.rates = np.asarray(self.rates, dtype=float)[order]
        if len(self.terms) == 0:
            raise ValueError("Yield curve must contain at least one term.")

    @classmethod
    def fixed(cls, rate: float) -> "YieldCurve":
        return cls(np.array([0.0, 120.0]), np.array([rate, rate]), name=f"fixed_{rate:.4%}")

    def parallel_shift(self, shift: float) -> "YieldCurve":
        return YieldCurve(self.terms.copy(), self.rates + float(shift), name=f"{self.name}_shift_{shift:+.4%}")

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> "YieldCurve":
        path = Path(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        return cls.from_dataframe(df, name=name or path.stem)

    @classmethod
    def from_url(cls, url: str, name: str = "public_curve") -> "YieldCurve":
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return cls.from_dataframe(pd.read_csv(StringIO(response.text)), name=name)

    @classmethod
    def from_treasury_csv_url(cls, year: int, row: int = 0) -> "YieldCurve":
        url = TREASURY_DAILY_CSV_TEMPLATE.format(year=year)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return cls.from_dataframe(df.iloc[[row]], name=f"treasury_{year}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "custom") -> "YieldCurve":
        out = df.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        if {"term", "rate"}.issubset(out.columns):
            clean = pd.DataFrame(
                {
                    "term": pd.to_numeric(out["term"], errors="coerce"),
                    "rate": pd.to_numeric(out["rate"], errors="coerce"),
                }
            ).dropna()
            return cls(clean["term"].to_numpy(float), clean["rate"].to_numpy(float), name=name)

        row = out.iloc[0]
        terms: list[float] = []
        rates: list[float] = []
        for column, term in TREASURY_MATURITY_MAP.items():
            if column in out.columns and pd.notna(row[column]):
                terms.append(term)
                rates.append(float(row[column]) / 100.0)
        if not terms:
            raise ValueError("Curve data must have term/rate columns or recognizable Treasury maturity columns.")
        return cls(np.array(terms), np.array(rates), name=name)

    def spot_rate(self, term: float) -> float:
        return float(np.interp(term, self.terms, self.rates, left=self.rates[0], right=self.rates[-1]))

    def discount_factor(self, term: float) -> float:
        if term <= 0:
            return 1.0
        return 1.0 / ((1.0 + self.spot_rate(term)) ** term)
