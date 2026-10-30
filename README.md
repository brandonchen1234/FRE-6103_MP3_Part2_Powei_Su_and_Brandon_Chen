# Mini-Project 3: Simplified CDO Analysis

**FRE 6103 – Valuation for Financial Engineers** | Prof. David Shimko | Sep 2026

## **The underlying assets**

A bank has placed 10 speculative grade corporate bonds in a CDO it is looking to sell. Each bond is 5 years in duration, makes quarterly coupon payments at an annual coupon rate of 6%, and the annual default probability is 4%. The LGD is 60% for all bonds. The market YTM on the bonds is 9%, and the risk-free interest rate is 1% per year. The “correlation” between any two bonds’ default dates is equal to a constant of 0.20. The bonds’ face values are $10 MM each.

## **The CDO**

The structuring group has proposed two PACs for the CDO. The Class A tranche has an initial notional value of $20 MM and a coupon rate of 2% per year. The Class B tranche has a notional value of $10 MM and a coupon rate of 4% per year. The bank will retain the residuals as equity.

## **The Waterfall**

Collateral interest cash flows are used first to pay Class A interest, then Class B interest, with any remaining interest distributed to the bank as the residual equity holder. Recovery and principal cash flows are applied sequentially to Class A principal, then Class B principal, with any remaining amount distributed to the bank. There are no carry-forwards, and all cash flows are distributed as earned. Tranche coupons are calculated on beginning-of-quarter outstanding principal balances. Because the project description does not specify a detailed principal amortization schedule, this principal allocation is treated as a modeling assumption.

> *Suggestion: Compute promised cash flows of the collateral, adjust the cash flows for default times, and distribute the cash flows according to the waterfall.*

## **Tasks**

* **Task 1.** Build a simulation model of the aggregate quarterly cash flows of the portfolio, as a function of the quarterly cash flows. Be sure to correlate the simulated default dates.
* **Task 2.** Apply the waterfall rules to determine the cash flow patterns to each class.
* **Task 3.** Provide a statistical analysis of the cash flows (both the collateral and the classes) in a report suitable for a bank client. Be sure to include probability distributions of the outcomes, and key sensitivities to the parameters.
  


# Mini-Project 3B: CDO Valuation Using the Lintner Model

**FRE 6103 – Valuation for Financial Engineers** | Prof. David Shimko | Oct 2026

---

## The underlying assets

The collateral pool consists of the same 10 speculative-grade corporate bonds established in Part I. Each bond is 5 years in duration (20 quarterly periods), makes quarterly coupon payments at an annual coupon rate of 6%, and carries an annual default probability of 4% with a Loss Given Default (LGD) of 60% (40% recovery rate upon default). The market YTM on the bonds is 9% (or 6% under par/yield pricing), and the risk-free interest rate is 4% per year (1% per quarter). The "correlation" between any two bonds' default dates is equal to a constant of 0.20. The bonds' face values are $10 MM each ($100 MM total collateral pool).

## The CDO

For Part II, the capital structure of the CDO has been re-tranche sized to reflect an updated financing proposal. The Class A tranche has an initial notional value of $15 MM and a coupon rate of 4.5% per year. The Class B tranche has a notional value of $30 MM and a coupon rate of 5.5% per year. The bank will retain the remaining $55 MM notional as residual equity.

## The Waterfall

Collateral interest cash flows are used first to pay Class A interest, then Class B interest, with any remaining interest distributed to the bank as the residual equity holder. Recovery and principal cash flows are applied sequentially to Class A principal, then Class B principal, with any remaining amount distributed to the bank. There are no carry-forwards, and all cash flows are distributed as earned. Tranche coupons are calculated on beginning-of-quarter outstanding principal balances. Because the project description does not specify a detailed principal amortization schedule, this principal allocation is treated as a modeling assumption.

> *Suggestion: Value the CDO by separating the time value of money from credit risk. First compute expected present values of each tranche across scenarios discounted at the risk-free rate, then deduce the market price of risk from the underlying assets to determine certainty-equivalent tranche values using the Lintner model.*

## Tasks

* **Task 1.** Compute the scheduled cash flows (waterfall) for all tranches assuming zero defaults. Class A has $15 MM in principal (target rate 4.5%), Class B has $30 MM at target rate 5.5%, and the Residual retains $55 MM. Discount all scheduled cash flows at the 4% p.a. risk-free rate (1% per quarter).
* **Task 2.** Prepare 1,000 simulations from your static random numbers using moment matching (standardizing draws to have an exact sample mean of 0 and sample standard deviation of 1) to eliminate small-sample noise and optimize Monte Carlo performance.
* **Task 3.** Compute the PV of each tranche in each scenario at the risk-free rate (4% p.a.). Compute the 3x3 covariance matrix of the three tranches' cash flows, and compute a default indicator for future optimization of the structure.
* **Task 4.** Find the implicit cost of the risk using the underlying bond valuations, and calculate the cost per unit risk (both per unit of portfolio variance and standard deviation).
* **Task 5.** Value the tranches using the Lintner certainty-equivalent method:
  $$V_i = \mathbb{E}[PV_i(r_f)] - \lambda_{\text{var}} \cdot \text{Cov}(PV_i, PV_{\text{Pool}})$$
  Evaluate the resulting tranche valuations against their initial par notional balances.
* **Task 6. BONUS:** Optimize the structure to give the highest value to the residual, holding the Tranche A principal fixed at $15 MM, evaluating the trade-off between debt leverage, market premium over par ($V_B - P_B$), and the zero-default constraint.
