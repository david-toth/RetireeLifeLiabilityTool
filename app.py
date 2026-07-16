from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from retiree_life_pricer.engine import PricingEngine
from retiree_life_pricer.models import PricingAssumptions
from retiree_life_pricer.mortality import ImprovementScale, MortalityTable, sample_mortality_table
from retiree_life_pricer.premium import (
    AgeRatePerThousandModel,
    CurrentPremiumToTargetLossRatioModel,
    FlatAnnualPremiumModel,
    FlatRatePerThousandModel,
    ParticipantPremiumModel,
    TargetLossRatioPremiumModel,
)
from retiree_life_pricer.reduction import (
    ReductionSchedules,
    annual_stepdown_rule,
    fixed_amount_by_age_rule,
    fixed_percent_by_age_rule,
    level_schedule,
    monthly_stepdown_rule,
)
from retiree_life_pricer.soa import MP_SCALES, PRI2012_TABLES, load_mp_scale, load_pri2012_table
from retiree_life_pricer.yield_curve import YieldCurve


st.set_page_config(page_title="Retiree Life Liability Tool", layout="wide")


DISCLAIMER_PARAGRAPHS = [
    (
        "This application provides preliminary estimates of retiree life insurance liabilities and premiums based "
        "on user-provided data, assumptions, and calculation settings. It is intended solely as an analytical "
        "support tool for its documented purpose and does not constitute an actuarial opinion, actuarial report, "
        "financial advice, or other professional advice."
    ),
    (
        "The application does not validate inputs for completeness, accuracy, consistency, or reasonableness. "
        "The user is solely responsible for reviewing and confirming all data, assumptions, and calculation settings."
    ),
    (
        "Results may be affected by inaccurate or incomplete inputs, inappropriate assumptions, methodological "
        "limitations, coding errors, software changes, or other defects. All results must be independently reviewed "
        "and verified by a qualified professional before use. The results should not be relied upon for financial "
        "reporting, pricing, transaction execution, regulatory compliance, legal, tax, accounting, actuarial, or "
        "other decision-making purposes."
    ),
    (
        "Any assumptions, methodologies, calculation parameters, limitations, or default settings not expressly "
        "displayed in the application are contained in the source code. The user is responsible for reviewing the "
        "source code in its entirety and understanding the application’s calculations, assumptions, intended use, "
        "and limitations before using, distributing, or relying upon any output."
    ),
    (
        "Use of this application and its outputs is entirely at the user’s own risk. No representation or warranty, "
        "express or implied, is made regarding the accuracy, completeness, reliability, or suitability of the "
        "application or its results."
    ),
]


def disclaimer_markdown() -> str:
    return "## Disclaimer\n\n" + "\n\n".join(DISCLAIMER_PARAGRAPHS)


def disclaimer_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [{"paragraph": index + 1, "disclaimer": paragraph} for index, paragraph in enumerate(DISCLAIMER_PARAGRAPHS)]
    )


@st.cache_data(show_spinner=False)
def parse_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def read_table(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return parse_uploaded_table(uploaded_file.name, uploaded_file.getvalue())


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def cached_pri2012_table(key: str) -> MortalityTable:
    return load_pri2012_table(key)


@st.cache_data(show_spinner=False)
def cached_mp_scale(key: str) -> ImprovementScale:
    return load_mp_scale(key)


def catalog_options(catalog) -> dict[str, str]:
    return {item.label: key for key, item in catalog.items()}


def default_cohort_assumptions(default_table: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cohort": "union", "mortality_table": default_table, "mortality_multiplier": 1.00},
            {"cohort": "non-union", "mortality_table": default_table, "mortality_multiplier": 1.00},
        ]
    )


def liability_pv_under_shift(
    cashflows: pd.DataFrame,
    curve: YieldCurve,
    assumptions: PricingAssumptions,
    shift: float,
) -> float:
    benefit_offsets = {"mid_year": 0.5, "end_of_year": 1.0}
    shifted_curve = curve.parallel_shift(shift)
    offset = benefit_offsets[assumptions.benefit_timing]
    terms = cashflows["projection_year"].to_numpy(float) - 1.0 + offset
    discount_factors = [shifted_curve.discount_factor(term) for term in terms]
    return float((cashflows["death_benefit_cashflow"].to_numpy(float) * discount_factors).sum())


def liability_interest_sensitivity(
    cashflows: pd.DataFrame,
    curve: YieldCurve,
    assumptions: PricingAssumptions,
    shock_bps: float,
) -> pd.DataFrame:
    shock = float(shock_bps) / 10000.0
    base_pv = liability_pv_under_shift(cashflows, curve, assumptions, 0.0)
    if shock <= 0.0 or base_pv == 0.0:
        return pd.DataFrame(
            [
                {
                    "measure": "Death Benefit Liability",
                    "pv": base_pv,
                    "shock_bps": shock_bps,
                    "effective_duration": 0.0,
                    "effective_convexity": 0.0,
                }
            ]
        )
    pv_up = liability_pv_under_shift(cashflows, curve, assumptions, shock)
    pv_down = liability_pv_under_shift(cashflows, curve, assumptions, -shock)
    return pd.DataFrame(
        [
            {
                "measure": "Death Benefit Liability",
                "pv": base_pv,
                "shock_bps": shock_bps,
                "pv_rate_up": pv_up,
                "pv_rate_down": pv_down,
                "effective_duration": (pv_down - pv_up) / (2.0 * base_pv * shock),
                "effective_convexity": (pv_down + pv_up - 2.0 * base_pv) / (base_pv * shock * shock),
            }
        ]
    )


st.title("Retiree Life Liability Tool")

with st.sidebar:
    pri_options = catalog_options(PRI2012_TABLES)
    mp_options = catalog_options(MP_SCALES)

    st.header("Input Files")
    participant_file = st.file_uploader("Participants", type=["csv", "xlsx", "xls"])
    mortality_file = st.file_uploader("Mortality table upload", type=["csv", "xlsx", "xls"])
    improvement_file = st.file_uploader("Improvement scale upload", type=["csv", "xlsx", "xls"])
    reduction_file = st.file_uploader("Reduction schedules", type=["csv", "xlsx", "xls"])
    premium_rate_file = st.file_uploader("Premium rates by age", type=["csv", "xlsx", "xls"])
    curve_file = st.file_uploader("Custom yield curve", type=["csv", "xlsx", "xls"])

    st.header("Valuation Date")
    valuation_date = st.date_input("Valuation date", value=date.today())

    st.header("Mortality")
    mortality_source = st.selectbox("Mortality source", ["SOA PRI-2012 dropdown", "Uploaded table", "Illustrative sample"])
    selected_pri_label = st.selectbox(
        "PRI-2012 mortality table",
        list(pri_options),
        index=list(pri_options).index("PRI-2012 Retiree - Amount-weighted"),
    )
    improvement_source = st.selectbox("Improvement source", ["SOA MP dropdown", "Uploaded scale", "None"])
    selected_mp_label = st.selectbox(
        "MP improvement scale",
        list(mp_options),
        index=list(mp_options).index("Scale MP-2021"),
    )
    mortality_improvement = st.checkbox("Apply mortality improvement", value=True)
    mortality_base_year = st.number_input(
        "Mortality base year",
        min_value=1900,
        max_value=2200,
        value=2012,
        step=1,
    )
    cohort_editor = st.data_editor(
        default_cohort_assumptions(selected_pri_label),
        column_config={
            "mortality_table": st.column_config.SelectboxColumn(
                "mortality_table",
                options=list(pri_options),
                required=True,
            ),
            "mortality_multiplier": st.column_config.NumberColumn(
                "mortality_multiplier",
                min_value=0.0,
                step=0.01,
                format="%.4f",
            ),
        },
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
    )

    st.header("Discount Rate")
    curve_type = st.selectbox("Discount basis", ["Fixed rate", "Custom uploaded curve"])
    fixed_rate = st.number_input("Fixed annual effective rate", value=0.045, step=0.001, format="%.4f")

    st.header("Premiums")
    premium_basis = st.selectbox(
        "Premium basis",
        [
            "Participant annual_premium column",
            "Flat annual premium amount",
            "Flat rate per 1,000 coverage",
            "Age rate per 1,000 coverage",
            "Target loss ratio",
            "Current premium graded to target loss ratio",
        ],
    )
    flat_annual_premium = st.number_input("Flat annual premium amount", min_value=0.0, value=500.0, step=25.0)
    flat_rate_per_1000 = st.number_input("Flat rate per 1,000", min_value=0.0, value=8.00, step=0.25, format="%.4f")
    target_loss_ratio_pct = st.number_input("Target loss ratio", min_value=0.0001, max_value=500.0, value=80.0, step=1.0, format="%.4f")
    current_premium_trend_pct = st.number_input("Current premium annual trend", value=0.0, step=0.25, format="%.4f")
    current_premium_grade_years = st.number_input(
        "Years to grade to target LR",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
    )

    st.header("Other")
    projection_years = st.slider("Projection years", min_value=1, max_value=100, value=80)
    default_reduction_schedule_id = st.text_input("Default reduction schedule", value="default")
    default_premium_end_age = st.number_input("Default premium end age", min_value=0.0, max_value=130.0, value=120.0)
    run_requested = st.button("Run valuation", type="primary", width="stretch")

tabs = st.tabs([
    "Summary",
    "Individual Results",
    "Cashflow Summary",
    "Detailed Liability Results",
    "Stepdown Builder",
])

participants_df = read_table(participant_file)
pricing_error = None
cashflows = None
summary = None
annual_by_cohort = None
annual_total = None
annual = None
total = None
interest_sensitivity = None
clean_cohorts = None
premium_assumptions = pd.DataFrame(
    [
        {
            "premium_basis": premium_basis,
            "flat_annual_premium": flat_annual_premium,
            "flat_rate_per_1000": flat_rate_per_1000,
            "target_loss_ratio": target_loss_ratio_pct / 100.0,
            "current_premium_trend": current_premium_trend_pct / 100.0,
            "current_premium_grade_years": int(current_premium_grade_years),
            "mortality_base_year": int(mortality_base_year),
            "valuation_year": valuation_date.year,
        }
    ]
)

if run_requested:
    if participants_df is None:
        st.session_state["valuation_error"] = "Upload a participant file before running the valuation."
        st.session_state.pop("valuation_results", None)
    else:
        try:
            with st.spinner("Running valuation..."):
                if mortality_source == "SOA PRI-2012 dropdown":
                    mortality = cached_pri2012_table(pri_options[selected_pri_label])
                elif mortality_source == "Uploaded table" and mortality_file:
                    mortality = MortalityTable(read_table(mortality_file), name="uploaded")
                else:
                    mortality = sample_mortality_table()

                if improvement_source == "SOA MP dropdown":
                    improvement = cached_mp_scale(mp_options[selected_mp_label]) if mortality_improvement else None
                elif improvement_source == "Uploaded scale" and improvement_file:
                    improvement = ImprovementScale(read_table(improvement_file), name="uploaded")
                else:
                    improvement = None

                reductions = ReductionSchedules(read_table(reduction_file)) if reduction_file else level_schedule()
                curve = (
                    YieldCurve.from_dataframe(read_table(curve_file), name="uploaded_curve")
                    if curve_type == "Custom uploaded curve" and curve_file
                    else YieldCurve.fixed(float(fixed_rate))
                )
                if premium_basis == "Participant annual_premium column":
                    premium_model = ParticipantPremiumModel()
                elif premium_basis == "Flat annual premium amount":
                    premium_model = FlatAnnualPremiumModel(float(flat_annual_premium))
                elif premium_basis == "Flat rate per 1,000 coverage":
                    premium_model = FlatRatePerThousandModel(float(flat_rate_per_1000))
                elif premium_basis == "Age rate per 1,000 coverage":
                    if premium_rate_file is None:
                        raise ValueError("Upload a premium rate table with age and rate_per_1000 columns.")
                    premium_model = AgeRatePerThousandModel(read_table(premium_rate_file))
                elif premium_basis == "Target loss ratio":
                    premium_model = TargetLossRatioPremiumModel(float(target_loss_ratio_pct) / 100.0)
                else:
                    premium_model = CurrentPremiumToTargetLossRatioModel(
                        target_loss_ratio=float(target_loss_ratio_pct) / 100.0,
                        annual_trend=float(current_premium_trend_pct) / 100.0,
                        grade_years=int(current_premium_grade_years),
                    )

                clean_cohorts = cohort_editor.copy()
                clean_cohorts["cohort"] = clean_cohorts["cohort"].astype(str).str.strip().str.lower()
                clean_cohorts["mortality_multiplier"] = pd.to_numeric(
                    clean_cohorts["mortality_multiplier"], errors="coerce"
                ).fillna(1.0)
                cohort_multipliers = dict(zip(clean_cohorts["cohort"], clean_cohorts["mortality_multiplier"]))
                cohort_mortality = None
                if mortality_source == "SOA PRI-2012 dropdown":
                    clean_cohorts["mortality_table"] = clean_cohorts["mortality_table"].fillna(selected_pri_label)
                    cohort_mortality = {
                        row["cohort"]: cached_pri2012_table(pri_options[row["mortality_table"]])
                        for _, row in clean_cohorts.iterrows()
                        if row["mortality_table"] in pri_options
                    }
                assumptions = PricingAssumptions(
                    valuation_date=valuation_date,
                    projection_years=projection_years,
                    default_reduction_schedule_id=default_reduction_schedule_id,
                    default_premium_end_age=default_premium_end_age,
                    mortality_improvement=mortality_improvement,
                    mortality_base_year=int(mortality_base_year),
                    cohort_mortality_multipliers=cohort_multipliers,
                )

                engine = PricingEngine(
                    mortality=mortality,
                    yield_curve=curve,
                    reductions=reductions,
                    improvement=improvement,
                    cohort_mortality=cohort_mortality,
                    premium_model=premium_model,
                )
                cashflows, summary = engine.project(participants_df, assumptions)
                annual_by_cohort = engine.annual_cohort_summary(cashflows)
                annual_total = annual_by_cohort[annual_by_cohort["cohort"] == "Total"]
                annual = cashflows.groupby("projection_year", as_index=False)[
                    ["death_benefit_cashflow", "premium_cashflow", "pv_death_benefit", "pv_future_premium"]
                ].sum()
                total = summary[["pv_death_benefit", "pv_future_premium", "net_pv_liability"]].sum()
                interest_sensitivity = liability_interest_sensitivity(
                    cashflows=cashflows,
                    curve=curve,
                    assumptions=assumptions,
                    shock_bps=100.0,
                )
                st.session_state["valuation_results"] = {
                    "cashflows": cashflows,
                    "summary": summary,
                    "annual_by_cohort": annual_by_cohort,
                    "annual_total": annual_total,
                    "annual": annual,
                    "total": total,
                    "interest_sensitivity": interest_sensitivity,
                    "clean_cohorts": clean_cohorts,
                    "premium_assumptions": premium_assumptions,
                }
                st.session_state["valuation_error"] = None
        except Exception as exc:
            st.session_state["valuation_error"] = str(exc)
            st.session_state.pop("valuation_results", None)

stored_results = st.session_state.get("valuation_results")
if stored_results is not None:
    cashflows = stored_results["cashflows"]
    summary = stored_results["summary"]
    annual_by_cohort = stored_results["annual_by_cohort"]
    annual_total = stored_results["annual_total"]
    annual = stored_results["annual"]
    total = stored_results["total"]
    interest_sensitivity = stored_results.get("interest_sensitivity")
    clean_cohorts = stored_results["clean_cohorts"]
    premium_assumptions = stored_results["premium_assumptions"]
pricing_error = st.session_state.get("valuation_error")


def report_ready() -> bool:
    if participants_df is None:
        st.info("Upload a participant file to run a valuation.")
        return False
    if pricing_error is not None:
        st.error(pricing_error)
        return False
    if stored_results is None:
        st.info("Click Run valuation in the sidebar to calculate results.")
        return False
    return True


with tabs[0]:
    if report_ready():
        st.subheader("Present Values")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PV Death Benefits", f"${total['pv_death_benefit']:,.0f}")
        col2.metric("PV Future Premiums", f"${total['pv_future_premium']:,.0f}")
        year_1_expected_benefits = annual.loc[
            annual["projection_year"] == 1, "death_benefit_cashflow"
        ].sum()
        col3.metric("Year 1 Expected Benefit Payments", f"${year_1_expected_benefits:,.0f}")
        loss_ratio = total["pv_death_benefit"] / total["pv_future_premium"] if total["pv_future_premium"] else 0.0
        col4.metric("PV Loss Ratio", f"{loss_ratio:.1%}")

        if interest_sensitivity is not None and not interest_sensitivity.empty:
            sensitivity_row = interest_sensitivity.iloc[0]
            st.subheader("Liability Interest Sensitivity")
            col1, col2 = st.columns(2)
            col1.metric("Effective Duration", f"{sensitivity_row['effective_duration']:,.2f}")
            col2.metric("Effective Convexity", f"{sensitivity_row['effective_convexity']:,.2f}")

        st.subheader("Expected Cashflows")
        annual_plot = annual.rename(
            columns={
                "projection_year": "Projection Year",
                "death_benefit_cashflow": "Death Benefits",
                "premium_cashflow": "Premiums",
            }
        )
        st.plotly_chart(
            px.bar(
                annual_plot,
                x="Projection Year",
                y=["Death Benefits", "Premiums"],
                labels={"value": "Expected Cashflow", "variable": "Cashflow Type"},
                barmode="group",
            ),
            width="stretch",
        )

        st.subheader("Data Summary")
        cohort_summary = (
            summary.groupby("cohort", as_index=False)
            .agg(
                count=("participant_id", "count"),
                pct_male=("sex", lambda sex: 100.0 * (sex.astype(str).str.upper() == "M").mean()),
                average_age=("valuation_age", "mean"),
                inforce_coverage=("inforce_coverage", "sum"),
                pv_liability=("pv_death_benefit", "sum"),
            )
            .sort_values("cohort")
        )
        st.dataframe(
            cohort_summary,
            column_config={
                "cohort": "Cohort",
                "count": st.column_config.NumberColumn("Count", format="%d"),
                "pct_male": st.column_config.NumberColumn("% Male", format="%.1f%%"),
                "average_age": st.column_config.NumberColumn("Average Age", format="%.1f"),
                "inforce_coverage": st.column_config.NumberColumn("In-force Coverage", format="$%,.0f"),
                "pv_liability": st.column_config.NumberColumn("PV Liability", format="$%,.0f"),
            },
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Export Results to Excel",
            data=to_excel_bytes(
                {
                    "summary": summary,
                    "cashflow_summary": annual_by_cohort.loc[
                        :,
                        [
                            "projection_year",
                            "calendar_year",
                            "cohort",
                            "death_benefit_cashflow",
                            "premium_cashflow",
                        ],
                    ],
                    "annual_total": annual_total,
                    "detailed_liability": cashflows,
                    "liability_sensitivity": interest_sensitivity if interest_sensitivity is not None else pd.DataFrame(),
                    "cohort_assumptions": clean_cohorts,
                    "premium_assumptions": premium_assumptions,
                    "disclaimer": disclaimer_dataframe(),
                }
            ),
            file_name="retiree_group_life_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("Disclaimer"):
            st.markdown(disclaimer_markdown())

with tabs[1]:
    if report_ready():
        st.subheader("Individual Results")
        st.dataframe(summary, width="stretch")

with tabs[2]:
    if report_ready():
        st.subheader("Cashflow Summary")
        cashflow_summary = (
            annual_by_cohort.loc[
                :,
                [
                    "projection_year",
                    "calendar_year",
                    "cohort",
                    "death_benefit_cashflow",
                    "premium_cashflow",
                ],
            ]
            .rename(
                columns={
                    "projection_year": "Projection Year",
                    "calendar_year": "Calendar Year",
                    "cohort": "Cohort",
                    "death_benefit_cashflow": "Expected Death Benefits",
                    "premium_cashflow": "Expected Premiums",
                }
            )
        )
        st.dataframe(
            cashflow_summary,
            column_config={
                "Projection Year": st.column_config.NumberColumn("Projection Year", format="%d"),
                "Calendar Year": st.column_config.NumberColumn("Calendar Year", format="%d"),
                "Cohort": "Cohort",
                "Expected Death Benefits": st.column_config.NumberColumn("Expected Death Benefits", format="$%,.0f"),
                "Expected Premiums": st.column_config.NumberColumn("Expected Premiums", format="$%,.0f"),
            },
            hide_index=True,
            width="stretch",
        )

with tabs[3]:
    if report_ready():
        st.subheader("Detailed Liability Results")
        st.dataframe(cashflows, width="stretch")

with tabs[4]:
    st.subheader("Reduction Schedule Builder")

    col1, col2, col3 = st.columns(3)
    builder_schedule_id = col1.text_input("Schedule ID", value="post65_monthly_stepdown")
    schedule_type = col2.selectbox(
        "Schedule type",
        ["Monthly percent stepdown", "Annual percent stepdown", "Fixed amount by age", "Fixed percent by age"],
    )
    builder_start_age = col3.number_input("Start age", min_value=0.0, max_value=130.0, value=65.0, step=0.25)

    if schedule_type == "Monthly percent stepdown":
        col_a, col_b = st.columns(2)
        monthly_reduction_pct = col_a.number_input(
            "Monthly reduction",
            min_value=0.0,
            max_value=100.0,
            value=2.5,
            step=0.1,
            format="%.4f",
        )
        minimum_factor_pct = col_b.number_input(
            "Ultimate factor",
            min_value=0.0,
            max_value=100.0,
            value=33.333333,
            step=0.1,
            format="%.6f",
        )
        builder_rule = monthly_stepdown_rule(
            schedule_id=builder_schedule_id,
            start_age=builder_start_age,
            monthly_reduction=monthly_reduction_pct / 100.0,
            minimum_factor=minimum_factor_pct / 100.0,
        )
    elif schedule_type == "Annual percent stepdown":
        col_a, col_b = st.columns(2)
        annual_reduction_pct = col_a.number_input(
            "Annual reduction",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            format="%.4f",
        )
        minimum_factor_pct = col_b.number_input(
            "Ultimate factor",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.5,
            format="%.4f",
        )
        builder_rule = annual_stepdown_rule(
            schedule_id=builder_schedule_id,
            start_age=builder_start_age,
            annual_reduction=annual_reduction_pct / 100.0,
            minimum_factor=minimum_factor_pct / 100.0,
        )
    elif schedule_type == "Fixed amount by age":
        amount_rows = st.data_editor(
            pd.DataFrame(
                [
                    {"age": 65, "amount": 100_000},
                    {"age": 70, "amount": 75_000},
                    {"age": 75, "amount": 50_000},
                ]
            ),
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
        )
        builder_rule = fixed_amount_by_age_rule(builder_schedule_id, amount_rows)
    else:
        factor_rows = st.data_editor(
            pd.DataFrame(
                [
                    {"age": 65, "factor": 1.00},
                    {"age": 70, "factor": 0.65},
                    {"age": 75, "factor": 0.50},
                ]
            ),
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
        )
        builder_rule = fixed_percent_by_age_rule(builder_schedule_id, factor_rows)

    candidate_participants = participants_df.copy() if participants_df is not None else None
    use_sample_test_case = (
        candidate_participants is None
        or candidate_participants.empty
        or "date_of_birth" not in candidate_participants.columns
    )
    if use_sample_test_case:
        if candidate_participants is not None and "date_of_birth" not in candidate_participants.columns:
            st.info("The uploaded participant data needs date_of_birth to test valuation-date-sensitive schedules. Using the sample test case here.")
        candidate_participants = pd.DataFrame(
            [
                {
                    "participant_id": "sample",
                    "sex": "M",
                    "date_of_birth": "1961-01-01",
                    "coverage_amount": 100_000,
                    "annual_premium": 0,
                    "premium_end_age": 120,
                    "reduction_schedule_id": builder_schedule_id,
                    "mortality_multiplier": 1.0,
                    "coverage_start_age": builder_start_age,
                    "cohort": "sample",
                }
            ]
        )

    participant_options = candidate_participants["participant_id"].astype(str).tolist()
    selected_test_id = st.selectbox("Test participant", participant_options)
    test_case = candidate_participants[
        candidate_participants["participant_id"].astype(str) == selected_test_id
    ].iloc[[0]].copy()
    test_case["reduction_schedule_id"] = builder_schedule_id
    if "coverage_start_age" not in test_case.columns:
        test_case["coverage_start_age"] = builder_start_age

    test_years = st.slider("Test output years", min_value=1, max_value=40, value=15)
    test_engine = PricingEngine(
        mortality=sample_mortality_table(),
        yield_curve=YieldCurve.fixed(float(fixed_rate)),
        reductions=ReductionSchedules(pd.DataFrame(columns=["schedule_id", "basis", "point", "factor"]), rules=builder_rule),
    )
    test_cashflows, _ = test_engine.project(
        test_case,
        PricingAssumptions(
            valuation_date=valuation_date,
            projection_years=test_years,
            mortality_improvement=False,
        ),
    )
    coverage_test = test_cashflows.loc[
        :,
        [
            "participant_id",
            "projection_year",
            "calendar_year",
            "attained_age",
            "coverage_amount",
            "benefit_factor",
            "benefit_amount",
        ],
    ].rename(columns={"benefit_amount": "annual_coverage_amount"})

    coverage_plot = coverage_test.rename(
        columns={
            "attained_age": "Attained Age",
            "benefit_factor": "Benefit Factor",
            "annual_coverage_amount": "Annual Coverage Amount",
        }
    )
    st.plotly_chart(
        px.line(
            coverage_plot,
            x="Attained Age",
            y=["Benefit Factor", "Annual Coverage Amount"],
            labels={"value": "Value", "variable": "Measure"},
        ),
        width="stretch",
    )

    st.subheader("Annual Coverage Test Output")
    st.dataframe(coverage_test, width="stretch")
    st.subheader("Reduction Schedule")
    st.dataframe(builder_rule, width="stretch")

    col_a, col_b = st.columns(2)
    col_a.download_button(
        "Download Reduction Schedule CSV",
        builder_rule.to_csv(index=False),
        file_name="reduction_schedules.csv",
    )
    col_b.download_button(
        "Download Test Output CSV",
        coverage_test.to_csv(index=False),
        file_name="reduction_schedule_test_output.csv",
    )
