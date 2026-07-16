# Retiree Life Liability Tool

Python and Streamlit toolkit for modeling retiree group life insurance liabilities under flexible benefit reduction schedules, mortality assumptions, yield curves, and premium runout assumptions.

## What It Does

- Defines a standard participant intake.
- Projects expected death benefit cashflows.
- Projects present value of future premiums payable to an insurer.
- Supports participant premiums, flat premiums, rates per 1,000 of coverage, age-banded rates, and target loss ratio premium projections.
- Supports fixed discount rates, custom spot curves, and public curve CSV imports.
- Supports custom mortality tables, SOA-style table exports, and improvement scales.
- Supports arbitrary reduction schedules by attained age, duration, or calendar year.
- Provides a Streamlit dashboard for uploads, assumption management, results, and exports.

Suggested GitHub repository name: `RetireeLifeLiabilityTool`.

## Disclaimer

This application provides preliminary estimates of retiree life insurance liabilities and premiums based on user-provided data, assumptions, and calculation settings. It is intended solely as an analytical support tool for its documented purpose and does not constitute an actuarial opinion, actuarial report, financial advice, or other professional advice.

The application does not validate inputs for completeness, accuracy, consistency, or reasonableness. The user is solely responsible for reviewing and confirming all data, assumptions, and calculation settings.

Results may be affected by inaccurate or incomplete inputs, inappropriate assumptions, methodological limitations, coding errors, software changes, or other defects. All results must be independently reviewed and verified by a qualified professional before use. The results should not be relied upon for financial reporting, pricing, transaction execution, regulatory compliance, legal, tax, accounting, actuarial, or other decision-making purposes.

Any assumptions, methodologies, calculation parameters, limitations, or default settings not expressly displayed in the application are contained in the source code. The user is responsible for reviewing the source code in its entirety and understanding the application’s calculations, assumptions, intended use, and limitations before using, distributing, or relying upon any output.

Use of this application and its outputs is entirely at the user’s own risk. No representation or warranty, express or implied, is made regarding the accuracy, completeness, reliability, or suitability of the application or its results.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

## Participant Intake

Required columns:

| column | description |
| --- | --- |
| `participant_id` | Unique participant identifier |
| `sex` | `M`, `F`, or another value supported by the mortality table |
| `date_of_birth` | Required date of birth; attained age is computed from the valuation date as `(valuation_date - date_of_birth) / 365.25` |
| `coverage_amount` | Current life insurance amount |

Optional columns:

| column | description |
| --- | --- |
| `annual_premium` | Annual premium payable to insurer |
| `premium_end_age` | Age when premium payments stop |
| `reduction_schedule_id` | Schedule key from uploaded reduction schedule |
| `mortality_multiplier` | Participant-specific mortality load |
| `coverage_start_age` | Age at which current coverage became effective |
| `cohort` | Cohort key for assumption overrides, such as `union` or `non-union` |

See `examples/participants.csv`.

## Premium Projection

The dashboard supports these premium bases:

- `Participant annual_premium column`: uses the participant file annual premium.
- `Flat annual premium amount`: applies one annual dollar premium to every surviving participant.
- `Flat rate per 1,000 coverage`: applies one rate to each participant's projected benefit amount.
- `Age rate per 1,000 coverage`: interpolates an uploaded age/rate table and applies it to projected benefit amount.
- `Target loss ratio`: sets expected premium cashflow equal to projected death benefit cashflow divided by the target loss ratio.
- `Current premium graded to target loss ratio`: trends the participant file annual premium, then linearly grades expected premium cashflow to the target-loss-ratio premium over the selected grade period.

Age-banded premium rate CSV:

```csv
age,rate_per_1000
65,8.00
70,12.00
75,18.00
```

## SOA PRI-2012 and MP Dropdowns

The dashboard includes dropdowns for sex-specific SOA MORT table pairs:

- PRI-2012 amount-weighted tables
- PRI.H-2012 headcount-weighted tables
- PRI-2012 juvenile tables
- MP-2014 through MP-2021 improvement scales

The app uses the SOA table identity for the participant sex: female participants use the female table identity and male participants use the male table identity. Cohorts can each select a different PRI-2012 base table, so a `union` cohort can use a different sex-specific base table pair than a `non-union` cohort. Custom uploaded mortality and improvement files remain available for locked-down production workflows.

Gender is always mapped from the participant `sex` field to the gender-specific base table. Cohort-level mortality multipliers can be entered in the dashboard and applied through the participant `cohort` column.

Fractional attained ages use UDD. For a projection year beginning at age `x + r`, the one-year death probability is derived from survival over the remaining `1-r` part of age `x` and the first `r` part of age `x+1`.

Mortality improvement uses the selected mortality base year, defaulting to `2012` for PRI-2012 tables, and the valuation year as the current year. Duration 0 mortality includes cumulative improvement from the base year up to the valuation year; later projection years continue generationally from there.

Participant files should not supply age as a substitute for `date_of_birth`; age is recalculated on every run from the selected valuation date so PVs respond to valuation date changes. The Excel reproduction is `=(valuation_date - date_of_birth) / 365.25`.

## Assumption Files

Mortality table CSV:

```csv
sex,age,qx
M,65,0.015
F,65,0.010
```

Improvement scale CSV:

```csv
sex,age,year,improvement
M,65,2026,0.012
```

Reduction schedule CSV:

```csv
schedule_id,basis,point,factor,type,start_age,monthly_reduction,annual_reduction,minimum_factor,age,amount
standard,age,65,1.00,,,,,,,
standard,age,70,0.65,,,,,,,
standard,age,75,0.50,,,,,,,
post65_monthly_stepdown,,,,monthly_stepdown,65,0.025,,0.3333333333,,
post65_annual_stepdown,,,,annual_stepdown,65,,0.10,0.50,,
fixed_amount_by_age,,,,fixed_amount_by_age,,,,,65,100000
fixed_amount_by_age,,,,fixed_amount_by_age,,,,,70,75000
fixed_amount_by_age,,,,fixed_amount_by_age,,,,,75,50000
fixed_percent_by_age,,,1.00,fixed_percent_by_age,,,,,65,
fixed_percent_by_age,,,0.65,fixed_percent_by_age,,,,,70,
fixed_percent_by_age,,,0.50,fixed_percent_by_age,,,,,75,
```

The same reduction schedule file can contain simple tabular schedules and standardized rule types. Rows with `basis`, `point`, and `factor` are treated as tabular schedules. Rows with `type` are treated as rule schedules.

The dashboard default reduction schedule is a portfolio-level override when that schedule ID exists in the uploaded reduction file. If the default ID is not present, the model falls back to each participant's `reduction_schedule_id`. If neither ID is known, no reduction is applied.

The Streamlit Reduction Schedule Builder can create and visually test standardized rule types:

- `monthly_stepdown`: reduces by a fixed percentage of initial coverage each month after `start_age`, subject to `minimum_factor`.
- `annual_stepdown`: reduces by a fixed percentage of initial coverage each year after `start_age`, subject to `minimum_factor`.
- `fixed_amount_by_age`: sets coverage to fixed dollar amounts by attained age using `age` and `amount` rows.
- `fixed_percent_by_age`: sets coverage to fixed multiples of initial coverage by attained age using `age` and `factor` rows.

Custom spot yield curve CSV:

```csv
term,rate
1,0.045
5,0.041
10,0.039
```

Rates are annual effective decimal spot rates. For public curves, the app can ingest a CSV URL if it contains maturity/rate fields or Treasury-style columns.
