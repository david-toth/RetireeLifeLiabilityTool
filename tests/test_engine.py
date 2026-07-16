from datetime import date

import pandas as pd

from retiree_life_pricer.engine import PricingEngine
from retiree_life_pricer.models import PricingAssumptions, validate_participants
from retiree_life_pricer.mortality import ImprovementScale, MortalityTable, sample_mortality_table
from retiree_life_pricer.premium import (
    AgeRatePerThousandModel,
    CurrentPremiumToTargetLossRatioModel,
    FlatRatePerThousandModel,
    TargetLossRatioPremiumModel,
)
from retiree_life_pricer.reduction import (
    ReductionSchedules,
    annual_stepdown_rule,
    fixed_amount_by_age_rule,
    fixed_percent_by_age_rule,
    monthly_stepdown_factor,
    monthly_stepdown_rule,
)
from retiree_life_pricer.yield_curve import YieldCurve


def test_engine_projects_benefits_and_premiums():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "annual_premium": 1_000,
                "premium_end_age": 67,
                "reduction_schedule_id": "default",
                "cohort": "union",
            }
        ]
    )
    reductions = ReductionSchedules(
        pd.DataFrame(
            [
                {"schedule_id": "default", "basis": "age", "point": 65, "factor": 1.0},
                {"schedule_id": "default", "basis": "age", "point": 66, "factor": 0.5},
            ]
        )
    )
    engine = PricingEngine(sample_mortality_table(), YieldCurve.fixed(0.05), reductions)
    cashflows, summary = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=3),
    )

    assert len(cashflows) == 3
    assert summary.loc[0, "pv_death_benefit"] > 0
    assert summary.loc[0, "pv_future_premium"] > 0
    assert cashflows.loc[1, "benefit_amount"] == 50_000
    assert cashflows.loc[0, "cohort"] == "union"
    assert summary.loc[0, "sex"] == "M"
    assert summary.loc[0, "cohort"] == "union"
    assert summary.loc[0, "coverage_amount"] == 100_000
    assert summary.loc[0, "inforce_coverage"] == cashflows.loc[0, "benefit_amount"]


def test_cohort_multiplier_changes_projected_qx():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "cohort": "union",
            },
            {
                "participant_id": "B",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "cohort": "non-union",
            },
        ]
    )
    engine = PricingEngine(sample_mortality_table(), YieldCurve.fixed(0.05), ReductionSchedules(
        pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])
    ))
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(
            valuation_date=date(2026, 1, 1),
            projection_years=1,
            cohort_mortality_multipliers={"union": 1.2, "non-union": 0.8},
        ),
    )

    qx_by_id = dict(zip(cashflows["participant_id"], cashflows["qx"].round(12)))
    assert qx_by_id["A"] > qx_by_id["B"]


def test_cohort_specific_mortality_tables_still_use_participant_sex():
    default_table = MortalityTable(
        pd.DataFrame(
            [
                {"sex": "M", "age": 65, "qx": 0.01},
                {"sex": "F", "age": 65, "qx": 0.02},
                {"sex": "M", "age": 66, "qx": 0.01},
                {"sex": "F", "age": 66, "qx": 0.02},
            ]
        ),
        name="default",
    )
    union_table = MortalityTable(
        pd.DataFrame(
            [
                {"sex": "M", "age": 65, "qx": 0.10},
                {"sex": "F", "age": 65, "qx": 0.20},
                {"sex": "M", "age": 66, "qx": 0.10},
                {"sex": "F", "age": 66, "qx": 0.20},
            ]
        ),
        name="union table",
    )
    non_union_table = MortalityTable(
        pd.DataFrame(
            [
                {"sex": "M", "age": 65, "qx": 0.30},
                {"sex": "F", "age": 65, "qx": 0.40},
                {"sex": "M", "age": 66, "qx": 0.30},
                {"sex": "F", "age": 66, "qx": 0.40},
            ]
        ),
        name="non-union table",
    )
    participants = pd.DataFrame(
        [
            {"participant_id": "M1", "sex": "M", "date_of_birth": "1961-01-01", "coverage_amount": 100_000, "cohort": "union"},
            {"participant_id": "F1", "sex": "F", "date_of_birth": "1961-01-01", "coverage_amount": 100_000, "cohort": "union"},
            {"participant_id": "M2", "sex": "M", "date_of_birth": "1961-01-01", "coverage_amount": 100_000, "cohort": "non-union"},
        ]
    )
    engine = PricingEngine(
        default_table,
        YieldCurve.fixed(0.05),
        ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])),
        cohort_mortality={"union": union_table, "non-union": non_union_table},
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=1),
    )

    qx_by_id = dict(zip(cashflows["participant_id"], cashflows["qx"].round(12)))
    table_by_id = dict(zip(cashflows["participant_id"], cashflows["mortality_table"]))
    assert qx_by_id == {"M1": 0.10, "F1": 0.20, "M2": 0.30}
    assert table_by_id == {"M1": "union table", "F1": "union table", "M2": "non-union table"}


def test_fractional_age_qx_uses_udd():
    table = MortalityTable(
        pd.DataFrame(
            [
                {"sex": "M", "age": 65, "qx": 0.10},
                {"sex": "M", "age": 66, "qx": 0.20},
                {"sex": "M", "age": 67, "qx": 0.30},
            ]
        )
    )

    expected = 1.0 - ((1.0 - 0.10) / (1.0 - 0.5 * 0.10)) * (1.0 - 0.5 * 0.20)

    assert table.qx("M", 65) == 0.10
    assert round(table.qx("M", 65.5), 12) == round(expected, 12)
    assert table.qx("M", 65.5) != 0.15


def test_date_of_birth_sets_age_from_valuation_date_using_36525_basis():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
            }
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])),
    )

    cashflows_2026, summary_2026 = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 1), projection_years=1),
    )
    cashflows_2027, summary_2027 = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2027, 1, 1), projection_years=1),
    )

    assert round(cashflows_2026.loc[0, "attained_age"], 6) == round(23741 / 365.25, 6)
    assert round(cashflows_2027.loc[0, "attained_age"], 6) == round(24106 / 365.25, 6)
    assert summary_2026.loc[0, "pv_death_benefit"] != summary_2027.loc[0, "pv_death_benefit"]

    cashflows_midyear, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 7, 1), projection_years=1),
    )
    assert round(cashflows_midyear.loc[0, "attained_age"], 6) == round(23922 / 365.25, 6)

    excel_check = pd.DataFrame(
        [
            {
                "participant_id": "Excel",
                "sex": "M",
                "date_of_birth": "1959-07-14",
                "coverage_amount": 100_000,
            }
        ]
    )
    cashflows_excel, _ = engine.project(
        excel_check,
        PricingAssumptions(valuation_date=date(2026, 7, 1), projection_years=1),
    )
    assert round(cashflows_excel.loc[0, "attained_age"], 5) == 66.96509


def test_age_only_participant_files_are_rejected():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "age": 65,
                "coverage_amount": 100_000,
            }
        ]
    )

    try:
        validate_participants(participants, valuation_date=date(2026, 1, 1))
    except ValueError as exc:
        assert "date_of_birth" in str(exc)
    else:
        raise AssertionError("age-only participant files should require date_of_birth")


def test_improvement_accumulates_from_base_year_to_valuation_year():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
            }
        ]
    )
    mortality = MortalityTable(
        pd.DataFrame(
            [
                {"sex": "M", "age": 60, "qx": 0.10},
                {"sex": "M", "age": 70, "qx": 0.10},
            ]
        )
    )
    improvement = ImprovementScale(
        pd.DataFrame(
            [
                {"sex": "M", "age": 60, "year": 2020, "improvement": 0.01},
                {"sex": "M", "age": 70, "year": 2020, "improvement": 0.01},
                {"sex": "M", "age": 60, "year": 2021, "improvement": 0.01},
                {"sex": "M", "age": 70, "year": 2021, "improvement": 0.01},
                {"sex": "M", "age": 60, "year": 2022, "improvement": 0.01},
                {"sex": "M", "age": 70, "year": 2022, "improvement": 0.01},
            ]
        )
    )
    engine = PricingEngine(
        mortality,
        YieldCurve.fixed(0.05),
        ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])),
        improvement=improvement,
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(
            valuation_date=date(2022, 1, 1),
            projection_years=2,
            mortality_base_year=2020,
        ),
    )

    assert round(cashflows.loc[0, "improvement_factor"], 12) == round(0.99**2, 12)
    assert round(cashflows.loc[0, "qx"], 12) == round(0.10 * 0.99**2, 12)
    assert round(cashflows.loc[1, "improvement_factor"], 12) == round(0.99**3, 12)


def test_annual_cohort_summary_includes_total_rows_and_total_cashflow():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "annual_premium": 100,
                "cohort": "union",
            },
            {
                "participant_id": "B",
                "sex": "F",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "annual_premium": 200,
                "cohort": "non-union",
            },
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=2),
    )
    annual = engine.annual_cohort_summary(cashflows)

    assert set(annual["cohort"]) == {"union", "non-union", "Total"}
    total_year_1 = annual[(annual["projection_year"] == 1) & (annual["cohort"] == "Total")].iloc[0]
    detail_year_1 = annual[(annual["projection_year"] == 1) & (annual["cohort"] != "Total")]
    assert total_year_1["death_benefit_cashflow"] == detail_year_1["death_benefit_cashflow"].sum()
    assert total_year_1["premium_cashflow"] == detail_year_1["premium_cashflow"].sum()
    assert total_year_1["total_cashflow"] == (
        total_year_1["death_benefit_cashflow"] - total_year_1["premium_cashflow"]
    )


def test_flat_rate_per_thousand_premiums_use_projected_benefit_amount():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
            }
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(
            pd.DataFrame(
                [
                    {"schedule_id": "default", "basis": "age", "point": 65, "factor": 1.0},
                    {"schedule_id": "default", "basis": "age", "point": 65.5, "factor": 1.0},
                    {"schedule_id": "default", "basis": "age", "point": 66, "factor": 0.5},
                ]
            )
        ),
        premium_model=FlatRatePerThousandModel(10.0),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=2),
    )

    assert cashflows.loc[0, "annual_premium"] == 1_000
    assert cashflows.loc[1, "annual_premium"] == 500
    assert cashflows.loc[0, "premium_basis"] == "flat_rate_per_1000"


def test_age_rate_per_thousand_premiums_interpolate_by_attained_age():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
            }
        ]
    )
    premium_model = AgeRatePerThousandModel(
        pd.DataFrame(
            [
                {"age": 65, "rate_per_1000": 8.0},
                {"age": 66, "rate_per_1000": 10.0},
            ]
        )
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])),
        premium_model=premium_model,
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 7, 1), projection_years=1),
    )

    assert 890 < cashflows.loc[0, "annual_premium"] < 910


def test_target_loss_ratio_premiums_are_implied_from_death_benefits():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
            }
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])),
        premium_model=TargetLossRatioPremiumModel(0.80),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 1), projection_years=2),
    )

    implied_loss_ratio = cashflows["death_benefit_cashflow"].sum() / cashflows["premium_cashflow"].sum()
    assert round(implied_loss_ratio, 12) == 0.80
    assert cashflows.loc[0, "premium_basis"] == "target_loss_ratio"


def test_current_premium_grades_to_target_loss_ratio():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "annual_premium": 1_000,
            }
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(pd.DataFrame([{"schedule_id": "default", "basis": "age", "point": 0, "factor": 1.0}])),
        premium_model=CurrentPremiumToTargetLossRatioModel(
            target_loss_ratio=0.80,
            annual_trend=0.0,
            grade_years=2,
        ),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=3),
    )

    year_1_expected = cashflows.loc[0, "survival_start"] * 1_000
    year_2_current = cashflows.loc[1, "survival_start"] * 1_000
    year_2_target = cashflows.loc[1, "death_benefit_cashflow"] / 0.80
    year_3_target = cashflows.loc[2, "death_benefit_cashflow"] / 0.80

    assert cashflows.loc[0, "premium_cashflow"] == year_1_expected
    assert round(cashflows.loc[1, "premium_cashflow"], 12) == round(0.5 * year_2_current + 0.5 * year_2_target, 12)
    assert round(cashflows.loc[2, "premium_cashflow"], 12) == round(year_3_target, 12)
    assert cashflows.loc[0, "premium_basis"] == "current_premium_to_target_loss_ratio"


def test_monthly_stepdown_rule_reduces_to_ultimate_factor():
    assert monthly_stepdown_factor(64.99, 65, 0.025, 1 / 3) == 1.0
    assert monthly_stepdown_factor(65, 65, 0.025, 1 / 3) == 1.0
    assert monthly_stepdown_factor(65 + 1 / 12, 65, 0.025, 1 / 3) == 0.975
    assert round(monthly_stepdown_factor(68, 65, 0.025, 1 / 3), 12) == round(1 / 3, 12)


def test_monthly_stepdown_rule_integrates_with_projection():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 90_000,
                "reduction_schedule_id": "post65_monthly_stepdown",
            }
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(
            pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]),
            rules=monthly_stepdown_rule("post65_monthly_stepdown", 65, 0.025, 1 / 3),
        ),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=4),
    )

    assert cashflows.loc[0, "benefit_factor"] == 1.0
    assert cashflows.loc[1, "benefit_factor"] == 0.7
    assert round(cashflows.loc[3, "benefit_factor"], 12) == round(1 / 3, 12)
    assert round(cashflows.loc[3, "benefit_amount"], 6) == 30_000


def test_combined_reduction_schedule_accepts_tabular_and_rule_rows():
    reductions = ReductionSchedules(
        pd.DataFrame(
            [
                {"schedule_id": "standard", "basis": "age", "point": 65, "factor": 1.0},
                {"schedule_id": "standard", "basis": "age", "point": 66, "factor": 0.5},
                {
                    "schedule_id": "monthly",
                    "type": "monthly_stepdown",
                    "start_age": 65,
                    "monthly_reduction": 0.025,
                    "minimum_factor": 1 / 3,
                },
            ]
        )
    )

    assert reductions.factor("standard", 66, 1, 2027, coverage_amount=100_000) == 0.5
    assert reductions.factor("monthly", 66, 1, 2027, coverage_amount=100_000) == 0.7


def test_example_reduction_schedule_file_runs_with_combined_format():
    reductions = ReductionSchedules.from_file("examples/reduction_schedules.csv")

    assert reductions.factor("standard", 70, 5, 2031, coverage_amount=100_000) == 0.65
    assert reductions.factor("post65_annual_stepdown", 66, 1, 2027, coverage_amount=100_000) == 0.9
    assert reductions.factor("fixed_amount_by_age", 70, 5, 2031, coverage_amount=100_000) == 0.75
    assert reductions.factor("fixed_percent_by_age", 70, 5, 2031, coverage_amount=100_000) == 0.65


def test_default_reduction_schedule_overrides_participant_schedule_when_valid():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "reduction_schedule_id": "participant_schedule",
            }
        ]
    )
    reductions = ReductionSchedules(
        pd.DataFrame(
            [
                {"schedule_id": "default_override", "basis": "age", "point": 65, "factor": 0.8},
                {"schedule_id": "participant_schedule", "basis": "age", "point": 65, "factor": 0.5},
            ]
        )
    )
    engine = PricingEngine(sample_mortality_table(), YieldCurve.fixed(0.05), reductions)
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(
            valuation_date=date(2026, 1, 2),
            projection_years=1,
            default_reduction_schedule_id="default_override",
        ),
    )

    assert cashflows.loc[0, "reduction_schedule_id"] == "default_override"
    assert cashflows.loc[0, "benefit_factor"] == 0.8


def test_participant_reduction_schedule_used_when_default_is_missing():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "reduction_schedule_id": "participant_schedule",
            }
        ]
    )
    reductions = ReductionSchedules(
        pd.DataFrame([{"schedule_id": "participant_schedule", "basis": "age", "point": 65, "factor": 0.5}])
    )
    engine = PricingEngine(sample_mortality_table(), YieldCurve.fixed(0.05), reductions)
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(
            valuation_date=date(2026, 1, 2),
            projection_years=1,
            default_reduction_schedule_id="missing_default",
        ),
    )

    assert cashflows.loc[0, "reduction_schedule_id"] == "participant_schedule"
    assert cashflows.loc[0, "benefit_factor"] == 0.5


def test_no_known_reduction_schedule_ids_keeps_benefits_level():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "reduction_schedule_id": "missing_participant",
            }
        ]
    )
    reductions = ReductionSchedules(pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]))
    engine = PricingEngine(sample_mortality_table(), YieldCurve.fixed(0.05), reductions)
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(
            valuation_date=date(2026, 1, 2),
            projection_years=1,
            default_reduction_schedule_id="missing_default",
        ),
    )

    assert cashflows.loc[0, "reduction_schedule_id"] == ""
    assert cashflows.loc[0, "benefit_factor"] == 1.0
    assert cashflows.loc[0, "benefit_amount"] == 100_000


def test_annual_stepdown_rule_integrates_with_projection():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "reduction_schedule_id": "annual_stepdown",
            }
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(
            pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]),
            rules=annual_stepdown_rule("annual_stepdown", 65, 0.10, 0.50),
        ),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=7),
    )

    assert cashflows.loc[0, "benefit_factor"] == 1.0
    assert cashflows.loc[1, "benefit_factor"] == 0.9
    assert cashflows.loc[5, "benefit_factor"] == 0.5
    assert cashflows.loc[6, "benefit_factor"] == 0.5


def test_fixed_amount_by_age_rule_integrates_with_projection():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "reduction_schedule_id": "fixed_amount_by_age",
            }
        ]
    )
    amount_rows = pd.DataFrame(
        [
            {"age": 65, "amount": 100_000},
            {"age": 66, "amount": 75_000},
            {"age": 67, "amount": 50_000},
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(
            pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]),
            rules=fixed_amount_by_age_rule("fixed_amount_by_age", amount_rows),
        ),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=4),
    )

    assert cashflows.loc[0, "benefit_amount"] == 100_000
    assert cashflows.loc[1, "benefit_amount"] == 75_000
    assert cashflows.loc[2, "benefit_amount"] == 50_000
    assert cashflows.loc[3, "benefit_amount"] == 50_000


def test_fixed_percent_by_age_rule_integrates_with_projection():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1961-01-01",
                "coverage_amount": 100_000,
                "reduction_schedule_id": "fixed_percent_by_age",
            }
        ]
    )
    factor_rows = pd.DataFrame(
        [
            {"age": 65, "factor": 1.00},
            {"age": 66, "factor": 0.75},
            {"age": 67, "factor": 0.50},
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(
            pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]),
            rules=fixed_percent_by_age_rule("fixed_percent_by_age", factor_rows),
        ),
    )
    cashflows, _ = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 1, 2), projection_years=4),
    )

    assert cashflows.loc[0, "benefit_amount"] == 100_000
    assert cashflows.loc[1, "benefit_amount"] == 75_000
    assert cashflows.loc[2, "benefit_amount"] == 50_000
    assert cashflows.loc[3, "benefit_amount"] == 50_000


def test_summary_inforce_coverage_reflects_reduction_in_effect_at_valuation():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "A",
                "sex": "M",
                "date_of_birth": "1959-07-14",
                "coverage_amount": 100_000,
                "reduction_schedule_id": "fixed_percent_by_age",
            }
        ]
    )
    engine = PricingEngine(
        sample_mortality_table(),
        YieldCurve.fixed(0.05),
        ReductionSchedules(
            pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]),
            rules=fixed_percent_by_age_rule(
                "fixed_percent_by_age",
                pd.DataFrame(
                    [
                        {"age": 65, "factor": 1.00},
                        {"age": 66, "factor": 0.75},
                    ]
                ),
            ),
        ),
    )
    _, summary = engine.project(
        participants,
        PricingAssumptions(valuation_date=date(2026, 7, 1), projection_years=1),
    )

    assert summary.loc[0, "coverage_amount"] == 100_000
    assert summary.loc[0, "inforce_coverage"] == 75_000
