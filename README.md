# LendScope: End-to-End Credit Risk & Lending Decision Pipeline

LendScope is an institutional-grade, production-ready credit risk modeling and decisioning system. Built using a 360-degree approach from raw data ingestion to regulatory stress testing, fairness auditing, and API deployment, this project provides a comprehensive template for modern retail lending risk management.

---

## Architecture & Project Phases

*   **Phase 1 — Data Engineering & Schema:** ETL pipeline converting wide-format UCI credit card data into a normalized relational schema (`dim_customer` and `fact_repayment`) stored in SQLite/PostgreSQL, featuring engineered utilization and payment ratios.
*   **Phase 2 — EDA & Statistical Analysis:** Hypothesis testing and bootstrapping 95% confidence intervals on effect sizes, confirming recent repayment status (`PAY_0`) as the dominant risk separator.
*   **Phase 3 — Probability of Default (PD) Model:** Logistic regression baseline alongside a non-linear LightGBM classifier achieving a robust **0.779 AUC**.
*   **Phase 4 — Expected Loss & Regulatory Modelling:** Portfolio-level Expected Loss calculation ($741M base) utilizing an industry-standard 85% LGD, coupled with a macroeconomic downturn stress test (+1 delinquency tier shift doubling losses to $1.47B).
*   **Phase 5 — Credit Policy Simulator:** Automated sweep of PD decision thresholds plotting approval rates vs. loss rates, segmented by credit limit tiers.
*   **Phase 6 — Explainability & Governance:** Reliability diagrams verifying model calibration alongside SHAP waterfall plots for individual adverse action notices, accompanied by production governance caveats.
*   **Phase 7 — Responsible-Lending / Fairness Check:** Demographic parity audit across age bands, gender, and education to detect disparate impact and justify dynamic threshold adjustments.
*   **Phase 8 — Deployment:** FastAPI containerized decisioning endpoint delivering real-time PD scoring, expected loss computation, and top risk driver extraction via SHAP.

---

## Tech Stack

*   **Language:** Python 3.10+
*   **Data & Modeling:** Pandas, NumPy, Scikit-Learn, LightGBM, Statsmodels, Scipy, SHAP
*   **Database & ETL:** SQLAlchemy, SQLite / PostgreSQL, DBeaver
*   **Deployment:** FastAPI, Uvicorn, Docker, Pydantic

---

## Quickstart & Installation

1. **Clone the repository and install dependencies:**
   ```powershell
   git clone [https://github.com/nmakoui/lendscope.git](https://github.com/nmakoui/lendscope.git)
   cd lendscope
   pip install -r requirements.txt