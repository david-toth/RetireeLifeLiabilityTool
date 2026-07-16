from __future__ import annotations

import csv
from dataclasses import dataclass
from html import unescape
from io import StringIO

import pandas as pd
import requests

from .mortality import ImprovementScale, MortalityTable


SOA_EXPORT_URL = "https://mort.soa.org/Export.aspx?Type=csv&TableIdentity={table_id}"
SOA_SEARCH_URL = "https://mort.soa.org/WebService.asmx/GetListOfTables"


@dataclass(frozen=True)
class SexSpecificTable:
    key: str
    label: str
    female_table_id: int
    male_table_id: int
    description: str

    def table_id_for_sex(self, sex: str) -> int:
        return self.female_table_id if str(sex).strip().upper().startswith("F") else self.male_table_id


_AMOUNT_PRI_CLASSES = [
    ("employee", "Employee"),
    ("retiree", "Retiree"),
    ("contingent_survivor", "Contingent Survivor"),
    ("disabled_retiree", "Disabled Retiree"),
    ("employee_bottom_quartile", "Employee Bottom Quartile"),
    ("retiree_bottom_quartile", "Retiree Bottom Quartile"),
    ("employee_top_quartile", "Employee Top Quartile"),
    ("retiree_top_quartile", "Retiree Top Quartile"),
    ("employee_blue_collar", "Employee Blue Collar"),
    ("retiree_blue_collar", "Retiree Blue Collar"),
    ("contingent_survivor_blue_collar", "Contingent Survivor Blue Collar"),
    ("employee_white_collar", "Employee White Collar"),
    ("retiree_white_collar", "Retiree White Collar"),
    ("contingent_survivor_white_collar", "Contingent Survivor White Collar"),
    ("nondisabled_annuitant", "Nondisabled Annuitant"),
    ("nondisabled_annuitant_blue_collar", "Nondisabled Annuitant Blue Collar"),
    ("nondisabled_annuitant_white_collar", "Nondisabled Annuitant White Collar"),
]


def _build_pri_catalog() -> dict[str, SexSpecificTable]:
    catalog: dict[str, SexSpecificTable] = {}
    for i, (slug, label) in enumerate(_AMOUNT_PRI_CLASSES):
        female_id = 3531 + i * 2
        male_id = female_id + 1
        key = f"pri2012_amount_{slug}"
        catalog[key] = SexSpecificTable(
            key=key,
            label=f"PRI-2012 {label} - Amount-weighted",
            female_table_id=female_id,
            male_table_id=male_id,
            description=f"SOA PRI-2012 amount-weighted {label} mortality table.",
        )
    for i, (slug, label) in enumerate(_AMOUNT_PRI_CLASSES):
        female_id = 3565 + i * 2
        male_id = female_id + 1
        key = f"pri2012_headcount_{slug}"
        catalog[key] = SexSpecificTable(
            key=key,
            label=f"PRI.H-2012 {label} - Headcount-weighted",
            female_table_id=female_id,
            male_table_id=male_id,
            description=f"SOA PRI-2012 headcount-weighted {label} mortality table.",
        )
    catalog["pri2012_amount_juvenile"] = SexSpecificTable(
        key="pri2012_amount_juvenile",
        label="PRI-2012 Juvenile",
        female_table_id=3599,
        male_table_id=3600,
        description="SOA PRI-2012 juvenile mortality table.",
    )
    return catalog


PRI2012_TABLES = _build_pri_catalog()


MP_SCALES: dict[str, SexSpecificTable] = {
    "mp2014": SexSpecificTable("mp2014", "Scale MP-2014", 3136, 3135, "SOA Scale MP-2014."),
    "mp2015": SexSpecificTable("mp2015", "Scale MP-2015", 3374, 3373, "SOA Scale MP-2015."),
    "mp2016": SexSpecificTable("mp2016", "Scale MP-2016", 3385, 3386, "SOA Scale MP-2016."),
    "mp2017": SexSpecificTable("mp2017", "Scale MP-2017", 3481, 3482, "SOA Scale MP-2017."),
    "mp2018": SexSpecificTable("mp2018", "Scale MP-2018", 3605, 3606, "SOA Scale MP-2018."),
    "mp2019": SexSpecificTable("mp2019", "Scale MP-2019", 3607, 3608, "SOA Scale MP-2019."),
    "mp2020": SexSpecificTable("mp2020", "Scale MP-2020", 3609, 3610, "SOA Scale MP-2020."),
    "mp2021": SexSpecificTable("mp2021", "Scale MP-2021", 3611, 3612, "SOA Scale MP-2021."),
}


def export_url(table_id: int) -> str:
    return SOA_EXPORT_URL.format(table_id=table_id)


def download_soa_csv(table_id: int) -> str:
    response = requests.get(export_url(table_id), timeout=30)
    response.raise_for_status()
    return response.text


def _data_rows(text: str) -> list[list[str]]:
    rows = list(csv.reader(StringIO(text)))
    for index, row in enumerate(rows):
        if row and row[0].strip().lower() == "row\\column":
            return rows[index:]
    raise ValueError("SOA export did not contain a Row\\Column data block.")


def parse_mortality_export(text: str, sex: str) -> pd.DataFrame:
    rows = _data_rows(text)
    parsed = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        age = pd.to_numeric(row[0], errors="coerce")
        qx = pd.to_numeric(row[1], errors="coerce")
        if pd.notna(age) and pd.notna(qx):
            parsed.append({"sex": sex, "age": float(age), "qx": float(qx)})
    if not parsed:
        raise ValueError("SOA mortality export did not contain usable age/qx rows.")
    return pd.DataFrame(parsed)


def parse_improvement_export(text: str, sex: str) -> pd.DataFrame:
    rows = _data_rows(text)
    header = rows[0]
    years = [pd.to_numeric(value, errors="coerce") for value in header[1:]]
    parsed = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        age = pd.to_numeric(row[0], errors="coerce")
        if pd.isna(age):
            continue
        for year, value in zip(years, row[1:]):
            improvement = pd.to_numeric(value, errors="coerce")
            if pd.notna(year) and pd.notna(improvement):
                parsed.append(
                    {
                        "sex": sex,
                        "age": float(age),
                        "year": int(year),
                        "improvement": float(improvement),
                    }
                )
    if not parsed:
        raise ValueError("SOA improvement export did not contain usable age/year rows.")
    return pd.DataFrame(parsed)


def load_pri2012_table(key: str) -> MortalityTable:
    selection = PRI2012_TABLES[key]
    female = parse_mortality_export(download_soa_csv(selection.female_table_id), "F")
    male = parse_mortality_export(download_soa_csv(selection.male_table_id), "M")
    return MortalityTable(pd.concat([female, male], ignore_index=True), name=selection.label)


def load_mp_scale(key: str) -> ImprovementScale:
    selection = MP_SCALES[key]
    female = parse_improvement_export(download_soa_csv(selection.female_table_id), "F")
    male = parse_improvement_export(download_soa_csv(selection.male_table_id), "M")
    return ImprovementScale(pd.concat([female, male], ignore_index=True), name=selection.label)


def clean_soa_table_name(value: str) -> str:
    return unescape(value).replace("\xa0", " ").replace("<br/>", " - ")

