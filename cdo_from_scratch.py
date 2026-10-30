import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# -----------------------------
# Project Parameters
# -----------------------------

# Make necessary directories
os.makedirs("distributions", exist_ok=True)
os.makedirs("sensitivity", exist_ok=True)

# Force the full table to print
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

N_SIM = 1000
N_BONDS = 10

YEARS = 5
QUARTERS_PER_YEAR = 4
N_QUARTERS = YEARS * QUARTERS_PER_YEAR

FACE_VALUE = 10_000_000
COUPON_RATE = 0.06

ANNUAL_DEFAULT_PROB = 0.04
LGD = 0.60
RECOVERY_RATE = 1 - LGD

DEFAULT_CORRELATION = 0.20

CLASS_A_NOTIONAL = 15_000_000
CLASS_A_COUPON = 0.045

CLASS_B_NOTIONAL = 30_000_000
CLASS_B_COUPON = 0.055

RISK_FREE_RATE = 0.04

SEED = 123

quarterly_bond_coupon = FACE_VALUE * COUPON_RATE / QUARTERS_PER_YEAR

quarterly_A_coupon = (
    CLASS_A_NOTIONAL * CLASS_A_COUPON / QUARTERS_PER_YEAR
)

quarterly_B_coupon = (
    CLASS_B_NOTIONAL * CLASS_B_COUPON / QUARTERS_PER_YEAR
)

# print("Bond quarterly coupon:", quarterly_bond_coupon)
# print("Class A quarterly coupon:", quarterly_A_coupon)
# print("Class B quarterly coupon:", quarterly_B_coupon)

# -----------------------------
# Step 2: Correlation Matrix
# -----------------------------

corr_matrix = np.full(
    (N_BONDS, N_BONDS),
    DEFAULT_CORRELATION
)

np.fill_diagonal(corr_matrix, 1.0)

# print("\nCorrelation matrix:")
# print(corr_matrix)

cholesky_matrix = np.linalg.cholesky(corr_matrix)

# print("\nCholesky matrix:")
# print(cholesky_matrix)


# -----------------------------
# Step 3: Generate Random Normals
# with Moment Matching
# -----------------------------

np.random.seed(SEED)

# Raw independent standard normals
independent_normals_raw = np.random.normal(
    size=(N_SIM, N_BONDS)
)

# Moment matching:
# force each bond column to have sample mean = 0
# and sample standard deviation = 1
independent_normals = (
    independent_normals_raw
    - independent_normals_raw.mean(axis=0)
) / independent_normals_raw.std(axis=0, ddof=1)

# Apply Cholesky correlation
correlated_normals = (
    independent_normals
    @ cholesky_matrix.T
)

# Re-standardize each correlated column
# so sample mean = 0 and std = 1
correlated_normals = (
    correlated_normals
    - correlated_normals.mean(axis=0)
) / correlated_normals.std(axis=0, ddof=1)

print("\nMoment Matching Verification")
print("----------------------------")

print(
    "Raw normal means:",
    np.round(independent_normals_raw.mean(axis=0), 4)
)

print(
    "Matched normal means:",
    np.round(independent_normals.mean(axis=0), 4)
)

print(
    "Matched normal std:",
    np.round(
        independent_normals.std(axis=0, ddof=1),
        4
    )
)

print(
    "Correlated normal means:",
    np.round(correlated_normals.mean(axis=0), 4)
)

print(
    "Correlated normal std:",
    np.round(
        correlated_normals.std(axis=0, ddof=1),
        4
    )
)

# print("\nCorrelated normals shape:")
# print(correlated_normals.shape)

sample_corr = np.corrcoef(
    correlated_normals,
    rowvar=False
)

print("\nSample correlation matrix:")
print(np.round(sample_corr, 2))

# -----------------------------
# Step 4: Convert Normal to Uniform
# -----------------------------

uniforms = norm.cdf(correlated_normals)

# print("\nUniforms shape:")
# print(uniforms.shape)

# print("\nFirst simulation - uniform values:")
# print(uniforms[0])

# -----------------------------
# Step 5: Convert Uniforms to Default Times
# -----------------------------

default_times = (
    np.log(1 - uniforms)
    / np.log(1 - ANNUAL_DEFAULT_PROB)
)

# print("\nDefault times shape:")
# print(default_times.shape)

print("\nFirst simulation - default times in years:")
print(np.round(default_times[0], 2))

# -----------------------------
# Step 6: Convert Default Times to Default Quarters
# -----------------------------

default_quarters = np.ceil(
    default_times * QUARTERS_PER_YEAR
).astype(int)

# print("\nFirst simulation - default quarters:")
# print(default_quarters[0])

# -----------------------------
# Step 7: Build Bond Cash Flows
# -----------------------------
recovery_amount = FACE_VALUE * RECOVERY_RATE

coupon_cash_flows = np.zeros(
    (N_SIM, N_BONDS, N_QUARTERS)
)

recovery_cash_flows = np.zeros(
    (N_SIM, N_BONDS, N_QUARTERS)
)

principal_cash_flows = np.zeros(
    (N_SIM, N_BONDS, N_QUARTERS)
)

for sim in range(N_SIM):

    for bond in range(N_BONDS):

        default_q = default_quarters[sim, bond]

        for q in range(1, N_QUARTERS + 1):

            # Bond defaults within the 5-year horizon
            if default_q <= N_QUARTERS:

                if q < default_q:
                    coupon_cash_flows[sim, bond, q - 1] = quarterly_bond_coupon

                elif q == default_q:
                    recovery_cash_flows[sim, bond, q - 1] = recovery_amount

            # Bond survives to maturity
            else:

                if q <= N_QUARTERS:
                    coupon_cash_flows[sim, bond, q - 1] = quarterly_bond_coupon

                if q == N_QUARTERS:
                    principal_cash_flows[sim, bond, q - 1] = FACE_VALUE

collateral_interest_cf = coupon_cash_flows.sum(axis=1)

collateral_recovery_cf = recovery_cash_flows.sum(axis=1)

collateral_principal_cf = principal_cash_flows.sum(axis=1)

total_collateral_cf = (
    collateral_interest_cf
    + collateral_recovery_cf
    + collateral_principal_cf
)

# print("\nCase 1 - Interest CF:")
# print(collateral_interest_cf[0])

# print("\nCase 1 - Recovery CF:")
# print(collateral_recovery_cf[0])

# print("\nCase 1 - Principal CF:")
# print(collateral_principal_cf[0])

# print("\nCase 1 - Total CF:")
# print(total_collateral_cf[0])

# -----------------------------
# Step 8: CDO Waterfall
# -----------------------------

class_A_interest_cf = np.zeros((N_SIM, N_QUARTERS))
class_B_interest_cf = np.zeros((N_SIM, N_QUARTERS))

class_A_principal_cf = np.zeros((N_SIM, N_QUARTERS))
class_B_principal_cf = np.zeros((N_SIM, N_QUARTERS))

equity_cf = np.zeros((N_SIM, N_QUARTERS))

class_A_balance = np.zeros((N_SIM, N_QUARTERS + 1))
class_B_balance = np.zeros((N_SIM, N_QUARTERS + 1))

class_A_balance[:, 0] = CLASS_A_NOTIONAL
class_B_balance[:, 0] = CLASS_B_NOTIONAL

for sim in range(N_SIM):

    for q in range(N_QUARTERS):

        # Beginning-of-quarter balances
        A_bal = class_A_balance[sim, q]
        B_bal = class_B_balance[sim, q]

        # -------------------------
        # 1. Interest Waterfall
        # -------------------------

        available_interest = collateral_interest_cf[sim, q]

        A_interest_due = A_bal * CLASS_A_COUPON / QUARTERS_PER_YEAR
        B_interest_due = B_bal * CLASS_B_COUPON / QUARTERS_PER_YEAR

        A_interest_paid = min(
            available_interest,
            A_interest_due
        )

        remaining_interest = available_interest - A_interest_paid

        B_interest_paid = min(
            remaining_interest,
            B_interest_due
        )

        remaining_interest -= B_interest_paid

        # -------------------------
        # 2. Principal Waterfall
        # -------------------------

        available_principal = (
            collateral_recovery_cf[sim, q]
            + collateral_principal_cf[sim, q]
        )

        A_principal_paid = min(
            available_principal,
            A_bal
        )

        remaining_principal = (
            available_principal
            - A_principal_paid
        )

        B_principal_paid = min(
            remaining_principal,
            B_bal
        )

        remaining_principal -= B_principal_paid

        # -------------------------
        # 3. Equity Gets Residual
        # -------------------------

        equity_paid = (
            remaining_interest
            + remaining_principal
        )

        # Save cash flows
        class_A_interest_cf[sim, q] = A_interest_paid
        class_B_interest_cf[sim, q] = B_interest_paid

        class_A_principal_cf[sim, q] = A_principal_paid
        class_B_principal_cf[sim, q] = B_principal_paid

        equity_cf[sim, q] = equity_paid

        # Update balances
        class_A_balance[sim, q + 1] = (
            A_bal - A_principal_paid
        )

        class_B_balance[sim, q + 1] = (
            B_bal - B_principal_paid
        )

class_A_total_cf = (
    class_A_interest_cf
    + class_A_principal_cf
)

class_B_total_cf = (
    class_B_interest_cf
    + class_B_principal_cf
)

# -----------------------------
# Part 2 - PV at Risk-Free Rate
# -----------------------------

quarterly_rf = RISK_FREE_RATE / QUARTERS_PER_YEAR

discount_factors = np.array([
    1 / (1 + quarterly_rf) ** t
    for t in range(1, N_QUARTERS + 1)
])

# ============================================================
# PART II STEP 1: Scheduled Zero-Default Waterfall
# ============================================================

# Scheduled collateral cash flow assuming no defaults
scheduled_collateral_cf = np.full(
    N_QUARTERS,
    N_BONDS * quarterly_bond_coupon
)

# Add all bond principal at maturity
scheduled_collateral_cf[-1] += (
    N_BONDS * FACE_VALUE
)

# Scheduled Class A cash flow
scheduled_A_cf = np.full(
    N_QUARTERS,
    CLASS_A_NOTIONAL
    * CLASS_A_COUPON
    / QUARTERS_PER_YEAR
)

scheduled_A_cf[-1] += CLASS_A_NOTIONAL

# Scheduled Class B cash flow
scheduled_B_cf = np.full(
    N_QUARTERS,
    CLASS_B_NOTIONAL
    * CLASS_B_COUPON
    / QUARTERS_PER_YEAR
)

scheduled_B_cf[-1] += CLASS_B_NOTIONAL

# Residual receives whatever is left
scheduled_equity_cf = (
    scheduled_collateral_cf
    - scheduled_A_cf
    - scheduled_B_cf
)

# Build table
scheduled_waterfall_df = pd.DataFrame({
    "Quarter": np.arange(1, N_QUARTERS + 1),
    "Collateral ($MM)": scheduled_collateral_cf / 1_000_000,
    "Class A ($MM)": scheduled_A_cf / 1_000_000,
    "Class B ($MM)": scheduled_B_cf / 1_000_000,
    "Residual ($MM)": scheduled_equity_cf / 1_000_000,
    "RF Discount Factor": discount_factors
})

print("\nScheduled Zero-Default Waterfall")
print("--------------------------------")
print(
    scheduled_waterfall_df.round(4).to_string(
        index=False
    )
)

# Scheduled PVs
scheduled_A_pv = (
    scheduled_A_cf @ discount_factors
)

scheduled_B_pv = (
    scheduled_B_cf @ discount_factors
)

scheduled_equity_pv = (
    scheduled_equity_cf @ discount_factors
)

scheduled_collateral_pv = (
    scheduled_collateral_cf @ discount_factors
)

print("\nScheduled Zero-Default PVs at 4% RF")
print("-----------------------------------")

print(
    f"Collateral scheduled PV: "
    f"${scheduled_collateral_pv / 1_000_000:.4f} MM"
)

print(
    f"Class A scheduled PV: "
    f"${scheduled_A_pv / 1_000_000:.4f} MM"
)

print(
    f"Class B scheduled PV: "
    f"${scheduled_B_pv / 1_000_000:.4f} MM"
)

print(
    f"Residual scheduled PV: "
    f"${scheduled_equity_pv / 1_000_000:.4f} MM"
)

pv_A = class_A_total_cf @ discount_factors
pv_B = class_B_total_cf @ discount_factors
pv_equity = equity_cf @ discount_factors

print("\nPV Summary at 4% Risk-Free Rate ($MM)")
print("--------------------------------------")
print(f"Class A mean PV: {pv_A.mean() / 1_000_000:.2f}")
print(f"Class B mean PV: {pv_B.mean() / 1_000_000:.2f}")
print(f"Equity mean PV:  {pv_equity.mean() / 1_000_000:.2f}")

# -----------------------------
# Part 2 - Covariance Matrix of Tranche PVs
# -----------------------------

pv_matrix = np.column_stack([
    pv_A,
    pv_B,
    pv_equity
])

cov_matrix = np.cov(
    pv_matrix,
    rowvar=False,
    ddof=1
)

print("\nCovariance Matrix of Tranche PVs ($MM^2)")
print("-----------------------------------------")
print(np.round(cov_matrix / 1_000_000**2, 4))

# -----------------------------
# Part 2 - Default Indicator
# -----------------------------

collateral_default_indicator = (
    default_quarters <= N_QUARTERS
).any(axis=1).astype(int)

default_scenario_count = collateral_default_indicator.sum()
default_scenario_probability = collateral_default_indicator.mean()

print("\nCollateral Bond Default Indicator")
print("---------------------------------")
print(
    f"Scenarios with at least one collateral bond default: "
    f"{default_scenario_count}"
)
print(
    f"Probability of at least one collateral bond default: "
    f"{default_scenario_probability:.2%}"
)

# -----------------------------
# Part 2 - Step 4: Market Value of Underlying Bonds
# -----------------------------

BOND_MARKET_YTM = 0.09
quarterly_ytm = BOND_MARKET_YTM / QUARTERS_PER_YEAR

bond_market_value = sum(
    quarterly_bond_coupon / (1 + quarterly_ytm) ** t
    for t in range(1, N_QUARTERS + 1)
)

bond_market_value += (
    FACE_VALUE / (1 + quarterly_ytm) ** N_QUARTERS
)

total_bond_market_value = N_BONDS * bond_market_value

print("\nUnderlying Bond Market Value")
print("----------------------------")
print(f"Market value per bond: ${bond_market_value / 1_000_000:.4f} MM")
print(f"Total market value:    ${total_bond_market_value / 1_000_000:.4f} MM")

pv_market = pv_A + pv_B + pv_equity

expected_collateral_pv = pv_market.mean()

print(
    f"Expected collateral PV @ risk-free rate: "
    f"${expected_collateral_pv / 1_000_000:.4f} MM"
)

total_risk_charge = (
    expected_collateral_pv
    - total_bond_market_value
)

market_variance = np.var(
    pv_market,
    ddof=1
)

cost_of_risk = (
    total_risk_charge
    / market_variance
)

print("\nLintner Risk Calibration")
print("------------------------")
print(
    f"Expected collateral PV @ RF: "
    f"${expected_collateral_pv / 1_000_000:.4f} MM"
)
print(
    f"Underlying bond market value: "
    f"${total_bond_market_value / 1_000_000:.4f} MM"
)
print(
    f"Total risk charge: "
    f"${total_risk_charge / 1_000_000:.4f} MM"
)
print(
    f"Portfolio variance: "
    f"{market_variance / 1_000_000**2:.4f} MM^2"
)
print(
    f"Cost of risk: "
    f"{cost_of_risk * 1_000_000:.6f} per $MM of variance"
)

# -----------------------------
# Part 2 - Step 5: Lintner Tranche Valuation
# -----------------------------

ones = np.ones(3)

# Covariance of each tranche PV with total market / collateral PV
cov_with_market = cov_matrix @ ones

# Risk charge allocated to each tranche
risk_charge_A = cost_of_risk * cov_with_market[0]
risk_charge_B = cost_of_risk * cov_with_market[1]
risk_charge_equity = cost_of_risk * cov_with_market[2]

# Mean PVs at the risk-free rate
mean_pv_A = pv_A.mean()
mean_pv_B = pv_B.mean()
mean_pv_equity = pv_equity.mean()

# Lintner values
lintner_value_A = mean_pv_A - risk_charge_A
lintner_value_B = mean_pv_B - risk_charge_B
lintner_value_equity = mean_pv_equity - risk_charge_equity

print("\nLintner Tranche Valuation")
print("-------------------------")

print(
    f"Class A covariance with market: "
    f"{cov_with_market[0] / 1_000_000**2:.4f} MM^2"
)
print(
    f"Class B covariance with market: "
    f"{cov_with_market[1] / 1_000_000**2:.4f} MM^2"
)
print(
    f"Equity covariance with market: "
    f"{cov_with_market[2] / 1_000_000**2:.4f} MM^2"
)

print()

print(
    f"Class A risk charge: "
    f"${risk_charge_A / 1_000_000:.4f} MM"
)
print(
    f"Class B risk charge: "
    f"${risk_charge_B / 1_000_000:.4f} MM"
)
print(
    f"Equity risk charge: "
    f"${risk_charge_equity / 1_000_000:.4f} MM"
)

print()

print(
    f"Class A Lintner value: "
    f"${lintner_value_A / 1_000_000:.4f} MM"
)
print(
    f"Class B Lintner value: "
    f"${lintner_value_B / 1_000_000:.4f} MM"
)
print(
    f"Equity Lintner value: "
    f"${lintner_value_equity / 1_000_000:.4f} MM"
)

print(
    f"Total Lintner value: "
    f"${(lintner_value_A + lintner_value_B + lintner_value_equity) / 1_000_000:.4f} MM"
)

# ============================================================
# BONUS: Optimize CDO Structure
# Hold Class A principal fixed at $15MM
# Vary Class B principal using the SAME simulated collateral CFs
# ============================================================

def value_structure(class_B_notional):

    # -----------------------------
    # 1. Initialize waterfall arrays
    # -----------------------------

    A_interest_cf = np.zeros((N_SIM, N_QUARTERS))
    B_interest_cf = np.zeros((N_SIM, N_QUARTERS))

    A_principal_cf = np.zeros((N_SIM, N_QUARTERS))
    B_principal_cf = np.zeros((N_SIM, N_QUARTERS))

    residual_cf = np.zeros((N_SIM, N_QUARTERS))

    A_balance = np.zeros((N_SIM, N_QUARTERS + 1))
    B_balance = np.zeros((N_SIM, N_QUARTERS + 1))

    A_balance[:, 0] = CLASS_A_NOTIONAL
    B_balance[:, 0] = class_B_notional

    # -----------------------------
    # 2. Waterfall
    # -----------------------------

    for sim in range(N_SIM):

        for q in range(N_QUARTERS):

            A_bal = A_balance[sim, q]
            B_bal = B_balance[sim, q]

            # ----- Interest waterfall -----

            available_interest = collateral_interest_cf[sim, q]

            A_interest_due = (
                A_bal
                * CLASS_A_COUPON
                / QUARTERS_PER_YEAR
            )

            B_interest_due = (
                B_bal
                * CLASS_B_COUPON
                / QUARTERS_PER_YEAR
            )

            A_interest_paid = min(
                available_interest,
                A_interest_due
            )

            remaining_interest = (
                available_interest
                - A_interest_paid
            )

            B_interest_paid = min(
                remaining_interest,
                B_interest_due
            )

            remaining_interest -= B_interest_paid

            # ----- Principal waterfall -----

            available_principal = (
                collateral_recovery_cf[sim, q]
                + collateral_principal_cf[sim, q]
            )

            A_principal_paid = min(
                available_principal,
                A_bal
            )

            remaining_principal = (
                available_principal
                - A_principal_paid
            )

            B_principal_paid = min(
                remaining_principal,
                B_bal
            )

            remaining_principal -= B_principal_paid

            # Residual gets everything left
            residual_paid = (
                remaining_interest
                + remaining_principal
            )

            # Store CFs
            A_interest_cf[sim, q] = A_interest_paid
            B_interest_cf[sim, q] = B_interest_paid

            A_principal_cf[sim, q] = A_principal_paid
            B_principal_cf[sim, q] = B_principal_paid

            residual_cf[sim, q] = residual_paid

            # Update balances
            A_balance[sim, q + 1] = (
                A_bal - A_principal_paid
            )

            B_balance[sim, q + 1] = (
                B_bal - B_principal_paid
            )

    # -----------------------------
    # 3. Total tranche cash flows
    # -----------------------------

    A_total_cf = A_interest_cf + A_principal_cf
    B_total_cf = B_interest_cf + B_principal_cf

    # -----------------------------
    # 4. PV at risk-free rate
    # -----------------------------

    pv_A_opt = A_total_cf @ discount_factors
    pv_B_opt = B_total_cf @ discount_factors
    pv_residual_opt = residual_cf @ discount_factors

    # -----------------------------
    # 5. Covariance matrix
    # -----------------------------

    pv_opt_matrix = np.column_stack([
        pv_A_opt,
        pv_B_opt,
        pv_residual_opt
    ])

    cov_opt = np.cov(
        pv_opt_matrix,
        rowvar=False,
        ddof=1
    )

    # Covariance with total collateral portfolio
    cov_market_opt = cov_opt @ np.ones(3)

    # -----------------------------
    # 6. Lintner values
    # -----------------------------

    risk_charge_opt = (
        cost_of_risk
        * cov_market_opt
    )

    mean_pvs = np.array([
        pv_A_opt.mean(),
        pv_B_opt.mean(),
        pv_residual_opt.mean()
    ])

    lintner_values = (
        mean_pvs
        - risk_charge_opt
    )

    # -----------------------------
# 7. CDO default / shortfall indicator
# -----------------------------

    A_interest_due_all = (
        A_balance[:, :-1]
        * CLASS_A_COUPON
        / QUARTERS_PER_YEAR
    )

    B_interest_due_all = (
        B_balance[:, :-1]
        * CLASS_B_COUPON
        / QUARTERS_PER_YEAR
    )

    A_interest_shortfall = np.maximum(
        A_interest_due_all - A_interest_cf,
        0
    )

    B_interest_shortfall = np.maximum(
        B_interest_due_all - B_interest_cf,
        0
    )

    A_default_indicator = (
        (A_interest_shortfall.sum(axis=1) > 1e-6)
        | (A_balance[:, -1] > 1e-6)
    )

    B_default_indicator = (
        (B_interest_shortfall.sum(axis=1) > 1e-6)
        | (B_balance[:, -1] > 1e-6)
    )

    CDO_default_indicator = (
        A_default_indicator
        | B_default_indicator
    )

    CDO_default_probability = (
        CDO_default_indicator.mean()
    )

    # -----------------------------
    # 8. Bank economic value
    # -----------------------------
    #
    # If A and B are issued at par:
    # Bank receives A principal + B principal in cash,
    # and keeps the residual.
    #

    # bank_value = (
    #     CLASS_A_NOTIONAL
    #     + class_B_notional
    #     + lintner_values[2]
    # )

    return {
    "B Notional": class_B_notional,
    "A Value": lintner_values[0],
    "B Value": lintner_values[1],
    "Residual Value": lintner_values[2],
    "CDO Default Probability": CDO_default_probability
    }

# -----------------------------
# BONUS - Grid Search
# -----------------------------

B_candidates = np.arange(
    0,
    71_000_000,
    1_000_000
)

optimization_results = []

for B_notional in B_candidates:

    result = value_structure(B_notional)

    optimization_results.append(result)

optimization_df = pd.DataFrame(
    optimization_results
)

# ============================================================
# FINAL BONUS FIGURE:
# CDO Capital Structure Optimization
# ============================================================

# ------------------------------------------------------------
# Extend optimization grid to 0–84 MM
# ------------------------------------------------------------

B_plot_candidates = np.arange(
    0,
    85_000_000,
    1_000_000
)

plot_results = [
    value_structure(B)
    for B in B_plot_candidates
]

plot_df = pd.DataFrame(plot_results)

# ------------------------------------------------------------
# Add useful derived quantities
# ------------------------------------------------------------

plot_df["B Notional MM"] = (
    plot_df["B Notional"] / 1_000_000
)

plot_df["Residual Value MM"] = (
    plot_df["Residual Value"] / 1_000_000
)

plot_df["B Premium MM"] = (
    plot_df["B Value"]
    - plot_df["B Notional"]
) / 1_000_000

plot_df["CDO Default Probability %"] = (
    plot_df["CDO Default Probability"] * 100
)

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, ax1 = plt.subplots(
    figsize=(11, 6.5)
)

# Residual Lintner Value
ax1.plot(
    plot_df["B Notional MM"],
    plot_df["Residual Value MM"],
    linewidth=2.5,
    label="Residual Lintner Value"
)

# 10x Class B premium for visibility
ax1.plot(
    plot_df["B Notional MM"],
    plot_df["B Premium MM"] * 10,
    linestyle="--",
    linewidth=2,
    label="10× Class B Premium over Par"
)

ax1.set_xlabel(
    "Class B Principal ($MM)"
)

ax1.set_ylabel(
    "Value ($MM)"
)

ax1.grid(
    True,
    alpha=0.3
)

# ------------------------------------------------------------
# Right axis: Default probability
# ------------------------------------------------------------

ax2 = ax1.twinx()

ax2.plot(
    plot_df["B Notional MM"],
    plot_df["CDO Default Probability %"],
    linestyle=":",
    linewidth=2.5,
    label="CDO Default Indicator Frequency"
)

ax2.set_ylabel(
    "CDO Default Indicator Frequency (%)"
)


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax1.set_title(
    "CDO Capital Structure Optimization\n"
    "(Class A Principal Fixed at $15 MM)"
)

# ------------------------------------------------------------
# Combined legend
# ------------------------------------------------------------

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    fontsize=9
)

# ------------------------------------------------------------
# Final layout
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    "distributions/lintner_optimization_final.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved final optimization chart: "
    "lintner_optimization_final.png"
)

# Convert dollar fields to $MM for display
optimization_display = optimization_df.copy()

for col in [
    "B Notional",
    "A Value",
    "B Value",
    "Residual Value"
]:

    optimization_display[col] = (
        optimization_display[col]
        / 1_000_000
    )

print("\nBONUS: CDO Structure Optimization")
print("----------------------------------")

print(
    optimization_display.round(4).to_string(
        index=False
    )
)

# -----------------------------
# BONUS - Find Maximum B Notional
# With Zero Simulated B Default
# -----------------------------

zero_default_df = optimization_df[
    optimization_df["CDO Default Probability"] == 0
].copy()

best_zero_default_idx = (
    zero_default_df["B Notional"].idxmax()
)

best_zero_default = (
    zero_default_df.loc[best_zero_default_idx]
)

print("\nBONUS: Maximum Zero-Default Class B Structure")
print("---------------------------------------------")

print(
    f"Maximum Class B Principal: "
    f"${best_zero_default['B Notional'] / 1_000_000:.2f} MM"
)

print(
    f"Class A Lintner Value: "
    f"${best_zero_default['A Value'] / 1_000_000:.4f} MM"
)

print(
    f"Class B Lintner Value: "
    f"${best_zero_default['B Value'] / 1_000_000:.4f} MM"
)

print(
    f"Residual Lintner Value: "
    f"${best_zero_default['Residual Value'] / 1_000_000:.4f} MM"
)

print(
    f"CDO Default Probability: "
    f"{best_zero_default['CDO Default Probability']:.2%}"
)

# -----------------------------
# BONUS - Fine Grid Search
# Around the zero-default boundary
# -----------------------------

B_fine_candidates = np.arange(
    27_000_000,
    28_100_000,
    10_000
)

fine_results = []

for B_notional in B_fine_candidates:
    fine_results.append(
        value_structure(B_notional)
    )

fine_df = pd.DataFrame(fine_results)

fine_display = fine_df.copy()

for col in [
    "B Notional",
    "A Value",
    "B Value",
    "Residual Value"
]:
    fine_display[col] = (
        fine_display[col] / 1_000_000
    )

print("\nBONUS: Fine Grid Search")
print("-----------------------")
print(
    fine_display[
        [
            "B Notional",
            "B Value",
            "Residual Value",
            "CDO Default Probability"
        ]
    ].round(4).to_string(index=False)
)

# Maximum B with zero simulated default
fine_zero_default = fine_df[
    fine_df["CDO Default Probability"] == 0
]

best_fine_idx = (
    fine_zero_default["B Notional"].idxmax()
)

best_fine = fine_zero_default.loc[
    best_fine_idx
]

print("\nBONUS: Refined Zero-Default Structure")
print("-------------------------------------")

print(
    f"Maximum Class B Principal: "
    f"${best_fine['B Notional'] / 1_000_000:.2f} MM"
)

print(
    f"Class B Lintner Value: "
    f"${best_fine['B Value'] / 1_000_000:.4f} MM"
)

print(
    f"Residual Lintner Value: "
    f"${best_fine['Residual Value'] / 1_000_000:.4f} MM"
)

print(
    f"CDO Default Probability: "
    f"{best_fine['CDO Default Probability']:.2%}"
)

# print("\nCase 1 - Class A Total CF:")
# print(class_A_total_cf[0])

# print("\nCase 1 - Class B Total CF:")
# print(class_B_total_cf[0])

# print("\nCase 1 - Equity CF:")
# print(equity_cf[0])

# print("\nCase 1 - Class A Balance:")
# print(class_A_balance[0])

# print("\nCase 1 - Class B Balance:")
# print(class_B_balance[0])

allocated_cf = (
    class_A_total_cf
    + class_B_total_cf
    + equity_cf
)

print(
    "\nCash flow conservation check:",
    np.allclose(
        allocated_cf,
        total_collateral_cf
    )
)

bond_cash_flows = (
    coupon_cash_flows
    + recovery_cash_flows
    + principal_cash_flows
)

def view_case(case_num):
    """
    Display detailed cash flows for one simulation case.
    case_num: 1 to 1000
    """

    if case_num < 1 or case_num > N_SIM:
        raise ValueError("Case number must be between 1 and 1000.")

    idx = case_num - 1

    # -----------------------------
    # 1. Individual Bond Cash Flows
    # -----------------------------

    bond_df = pd.DataFrame(
        bond_cash_flows[idx].T,
        index=[f"Q{i}" for i in range(1, N_QUARTERS + 1)],
        columns=[f"Bond {i}" for i in range(1, N_BONDS + 1)]
    )

    print("\n" + "=" * 100)
    print(f"CASE {case_num}: INDIVIDUAL BOND CASH FLOWS")
    print("=" * 100)

    print(
        bond_df.map(
            lambda x: f"{x:,.0f}"
        )
    )

    # -----------------------------
    # 2. Collateral + CDO Waterfall
    # -----------------------------

    waterfall_df = pd.DataFrame(
        {
            "Collateral Interest": collateral_interest_cf[idx],
            "Recovery": collateral_recovery_cf[idx],
            "Principal": collateral_principal_cf[idx],
            "Total Collateral": total_collateral_cf[idx],
            "Class A": class_A_total_cf[idx],
            "Class B": class_B_total_cf[idx],
            "Equity": equity_cf[idx],
        },
        index=[f"Q{i}" for i in range(1, N_QUARTERS + 1)]
    )

    print("\n" + "=" * 100)
    print(f"CASE {case_num}: CDO WATERFALL")
    print("=" * 100)

    print(
        waterfall_df.map(
            lambda x: f"{x:,.0f}"
        )
    )

# -----------------------------
# Step 9: Statistical Analysis
# -----------------------------

total_cf_collateral = total_collateral_cf.sum(axis=1)
total_cf_A = class_A_total_cf.sum(axis=1)
total_cf_B = class_B_total_cf.sum(axis=1)
total_cf_equity = equity_cf.sum(axis=1)

results_df = pd.DataFrame({
    "Collateral": total_cf_collateral,
    "Class A": total_cf_A,
    "Class B": total_cf_B,
    "Equity": total_cf_equity
})

stats = results_df.describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

print("\n" + "=" * 100)
print("TASK 3: TOTAL 5-YEAR CASH FLOW STATISTICS")
print("=" * 100)

stats_display = stats.copy()

for col in stats_display.columns:
    if col != "count":
        stats_display[col] = stats_display[col] / 1_000_000

print(stats_display.round(2))

# -----------------------------
# True Payment Shortfall Analysis
# -----------------------------

# Interest due each quarter based on beginning balance
A_interest_due = (
    class_A_balance[:, :-1]
    * CLASS_A_COUPON
    / QUARTERS_PER_YEAR
)

B_interest_due = (
    class_B_balance[:, :-1]
    * CLASS_B_COUPON
    / QUARTERS_PER_YEAR
)

# Unpaid interest
A_interest_shortfall = np.maximum(
    A_interest_due - class_A_interest_cf,
    0
)

B_interest_shortfall = np.maximum(
    B_interest_due - class_B_interest_cf,
    0
)

# Probability that any interest payment is missed
prob_A_interest_shortfall = np.mean(
    A_interest_shortfall.sum(axis=1) > 1e-6
)

prob_B_interest_shortfall = np.mean(
    B_interest_shortfall.sum(axis=1) > 1e-6
)

# Principal still unpaid after Q20
prob_A_principal_shortfall = np.mean(
    class_A_balance[:, -1] > 1e-6
)

prob_B_principal_shortfall = np.mean(
    class_B_balance[:, -1] > 1e-6
)

print("\nTrue Payment Shortfall Probabilities")
print("------------------------------------")
print(
    f"Class A interest shortfall: "
    f"{prob_A_interest_shortfall:.2%}"
)
print(
    f"Class B interest shortfall: "
    f"{prob_B_interest_shortfall:.2%}"
)
print(
    f"Class A principal shortfall: "
    f"{prob_A_principal_shortfall:.2%}"
)
print(
    f"Class B principal shortfall: "
    f"{prob_B_principal_shortfall:.2%}"
)

# -----------------------------
# Step 10: Probability Distributions
# -----------------------------

plt.figure()

plt.hist(
    total_cf_collateral / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Collateral Cash Flows")
plt.savefig("distributions/collateral_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()

plt.hist(
    total_cf_A / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Class A Cash Flows")
plt.savefig("distributions/class_A_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()

plt.hist(
    total_cf_B / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Class B Cash Flows")
plt.savefig("distributions/class_B_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()

plt.hist(
    total_cf_equity / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Equity Cash Flows")
plt.savefig("distributions/equity_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Step 11: Sensitivity Analysis
# -----------------------------

def run_scenario(pd_annual, lgd, rho, seed=123):

    np.random.seed(seed)

    recovery_rate = 1 - lgd
    recovery_amount = FACE_VALUE * recovery_rate

    # Correlation matrix
    corr_matrix = np.full((N_BONDS, N_BONDS), rho)
    np.fill_diagonal(corr_matrix, 1.0)

    L = np.linalg.cholesky(corr_matrix)

    # Independent normals -> moment matched -> correlated normals
    independent_normals_raw = np.random.normal(
    size=(N_SIM, N_BONDS)
    )

# Moment matching: force sample mean = 0 and std = 1
    independent_normals = (
        independent_normals_raw
        - independent_normals_raw.mean(axis=0)
    ) / independent_normals_raw.std(axis=0, ddof=1)

    # Apply Cholesky correlation
    correlated_normals = (
        independent_normals @ L.T
    )

    # Re-standardize correlated normals
    correlated_normals = (
        correlated_normals
        - correlated_normals.mean(axis=0)
    ) / correlated_normals.std(axis=0, ddof=1)

    # Normal -> Uniform
    uniforms = norm.cdf(correlated_normals)

    # Uniform -> default time
    default_times = (
        np.log(1 - uniforms)
        / np.log(1 - pd_annual)
    )

    default_quarters = np.ceil(
        default_times * QUARTERS_PER_YEAR
    ).astype(int)

    # Bond cash-flow components
    coupon_cf = np.zeros(
        (N_SIM, N_BONDS, N_QUARTERS)
    )

    recovery_cf = np.zeros(
        (N_SIM, N_BONDS, N_QUARTERS)
    )

    principal_cf = np.zeros(
        (N_SIM, N_BONDS, N_QUARTERS)
    )

    for sim in range(N_SIM):
        for bond in range(N_BONDS):

            dq = default_quarters[sim, bond]

            for q in range(1, N_QUARTERS + 1):

                if dq <= N_QUARTERS:

                    if q < dq:
                        coupon_cf[sim, bond, q - 1] = quarterly_bond_coupon

                    elif q == dq:
                        recovery_cf[sim, bond, q - 1] = recovery_amount

                else:

                    coupon_cf[sim, bond, q - 1] = quarterly_bond_coupon

                    if q == N_QUARTERS:
                        principal_cf[sim, bond, q - 1] = FACE_VALUE

    # Aggregate collateral cash flows
    collateral_interest = coupon_cf.sum(axis=1)
    collateral_recovery = recovery_cf.sum(axis=1)
    collateral_principal = principal_cf.sum(axis=1)

    collateral_total = (
        collateral_interest
        + collateral_recovery
        + collateral_principal
    )

    # Waterfall arrays
    class_A_interest_cf_s = np.zeros((N_SIM, N_QUARTERS))
    class_B_interest_cf_s = np.zeros((N_SIM, N_QUARTERS))

    class_A_principal_cf_s = np.zeros((N_SIM, N_QUARTERS))
    class_B_principal_cf_s = np.zeros((N_SIM, N_QUARTERS))

    equity_cf_s = np.zeros((N_SIM, N_QUARTERS))

    class_A_balance_s = np.zeros((N_SIM, N_QUARTERS + 1))
    class_B_balance_s = np.zeros((N_SIM, N_QUARTERS + 1))

    class_A_balance_s[:, 0] = CLASS_A_NOTIONAL
    class_B_balance_s[:, 0] = CLASS_B_NOTIONAL

    # Waterfall
    for sim in range(N_SIM):

        for q in range(N_QUARTERS):

            A_bal = class_A_balance_s[sim, q]
            B_bal = class_B_balance_s[sim, q]

            # Interest waterfall
            available_interest = collateral_interest[sim, q]

            A_interest_due = (
                A_bal * CLASS_A_COUPON / QUARTERS_PER_YEAR
            )

            B_interest_due = (
                B_bal * CLASS_B_COUPON / QUARTERS_PER_YEAR
            )

            A_interest_paid = min(
                available_interest,
                A_interest_due
            )

            remaining_interest = (
                available_interest
                - A_interest_paid
            )

            B_interest_paid = min(
                remaining_interest,
                B_interest_due
            )

            remaining_interest -= B_interest_paid

            # Principal waterfall
            available_principal = (
                collateral_recovery[sim, q]
                + collateral_principal[sim, q]
            )

            A_principal_paid = min(
                available_principal,
                A_bal
            )

            remaining_principal = (
                available_principal
                - A_principal_paid
            )

            B_principal_paid = min(
                remaining_principal,
                B_bal
            )

            remaining_principal -= B_principal_paid

            # Equity residual
            equity_paid = (
                remaining_interest
                + remaining_principal
            )

            class_A_interest_cf_s[sim, q] = A_interest_paid
            class_B_interest_cf_s[sim, q] = B_interest_paid

            class_A_principal_cf_s[sim, q] = A_principal_paid
            class_B_principal_cf_s[sim, q] = B_principal_paid

            equity_cf_s[sim, q] = equity_paid

            class_A_balance_s[sim, q + 1] = (
                A_bal - A_principal_paid
            )

            class_B_balance_s[sim, q + 1] = (
                B_bal - B_principal_paid
            )

    class_A_total_cf_s = (
        class_A_interest_cf_s
        + class_A_principal_cf_s
    )

    class_B_total_cf_s = (
        class_B_interest_cf_s
        + class_B_principal_cf_s
    )

    total_collateral_s = collateral_total.sum(axis=1)
    total_A_s = class_A_total_cf_s.sum(axis=1)
    total_B_s = class_B_total_cf_s.sum(axis=1)
    total_equity_s = equity_cf_s.sum(axis=1)

    return {
        "Collateral": total_collateral_s,
        "Class A": total_A_s,
        "Class B": total_B_s,
        "Equity": total_equity_s
    }


scenarios = {
    "Base": (0.04, 0.60, 0.20),
    "Low PD": (0.02, 0.60, 0.20),
    "High PD": (0.06, 0.60, 0.20),
    "Low LGD": (0.04, 0.40, 0.20),
    "High LGD": (0.04, 0.80, 0.20),
    "Low Corr": (0.04, 0.60, 0.10),
    "High Corr": (0.04, 0.60, 0.40),
}

sensitivity_rows = []

for name, (pd_val, lgd_val, rho_val) in scenarios.items():

    result = run_scenario(
        pd_annual=pd_val,
        lgd=lgd_val,
        rho=rho_val
    )

    sensitivity_rows.append({
    "Scenario": name,

    "Collateral Mean": result["Collateral"].mean() / 1_000_000,
    "Collateral Std": result["Collateral"].std(ddof=1) / 1_000_000,
    "Collateral 5%": np.percentile(result["Collateral"], 5) / 1_000_000,

    "Class A Mean": result["Class A"].mean() / 1_000_000,

    "Class B Mean": result["Class B"].mean() / 1_000_000,

    "Equity Mean": result["Equity"].mean() / 1_000_000,
    "Equity Std": result["Equity"].std(ddof=1) / 1_000_000,
    "Equity 5%": np.percentile(result["Equity"], 5) / 1_000_000,
})

sensitivity_df = pd.DataFrame(sensitivity_rows)

print("\n" + "=" * 100)
print("SENSITIVITY ANALYSIS ($MM)")
print("=" * 100)

print(sensitivity_df.round(2))

sensitivity_df.to_csv(
    "sensitivity/sensitivity_analysis.csv",
    index=False
)

# -----------------------------
# Step 12: Sensitivity Charts
# -----------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    sensitivity_df["Scenario"],
    sensitivity_df["Collateral 5%"]
)

plt.ylabel("Collateral 5th Percentile ($MM)")
plt.title("Sensitivity of Collateral Tail Cash Flow")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "sensitivity/sensitivity_collateral_5pct.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(10, 6))

plt.bar(
    sensitivity_df["Scenario"],
    sensitivity_df["Equity 5%"]
)

plt.ylabel("Equity 5th Percentile ($MM)")
plt.title("Sensitivity of Equity Tail Cash Flow")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "sensitivity/sensitivity_equity_5pct.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# while True:

#     user_input = input(
#         "\nEnter a case number (1-1000), or 'q' to quit: "
#     )

#     if user_input.lower() == "q":
#         break

#     try:
#         case_num = int(user_input)
#         view_case(case_num)

#     except ValueError:
#         print("Please enter a valid case number from 1 to 1000.")