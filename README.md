# LendScope: End-to-End Credit Risk & Lending Decision Pipeline

LendScope is an institutional-grade, production-ready credit risk modeling and decisioning system. Built using a 360-degree approach from raw data ingestion to regulatory stress testing, fairness auditing, and API deployment, this project provides a comprehensive template for modern retail lending risk management.

*(Note: tone/wording of this intro paragraph is unchanged from your draft - flagged separately as a style question, not a bug.)*

---

## Dataset

Built on the UCI "Default of Credit Card Clients" dataset (Yeh & Lien, 2009) - 30,000 credit card holders in **Taiwan**, snapshot April-September 2005. All monetary figures below (`LIMIT_BAL`, Expected Loss, stress-test losses) are in **New Taiwan Dollars (NT$)**, not USD.

---

## Architecture & Project Phases

*   **Phase 1 — Data Engineering & Schema:** ETL pipeline converting wide-format UCI credit card data into a normalized relational schema (`dim_customer` and `fact_repayment`) stored in SQLite, featuring engineered utilization and payment ratios.
*   **Phase 2 — EDA & Statistical Analysis:** Hypothesis testing and bootstrapping 95% confidence intervals on effect sizes, confirming recent repayment status (`PAY_0`) as the dominant risk separator.
*   **Phase 3 — Probability of Default (PD) Model:** Logistic regression baseline (0.700 AUC) alongside a non-linear LightGBM classifier achieving **0.779 AUC**, validated on a held-out test split.
*   **Phase 4 — Expected Loss & Regulatory Modelling:** Portfolio-level Expected Loss calculation (**NT$741M base**) utilizing an industry-standard 85% LGD, coupled with a macroeconomic downturn stress test (+1 delinquency tier shift doubling losses to **NT$1.47B**). Scored on the full live book, consistent with production scoring — not a held-out validation of the EL figures themselves (see Phase 3 for the genuine out-of-sample check).
*   **Phase 5 — Credit Policy Simulator:** Automated sweep of PD decision thresholds plotting approval rates vs. loss rates, segmented by credit limit tiers.
*   **Phase 6 — Explainability & Governance:** Reliability diagrams checking model calibration alongside SHAP waterfall plots for individual adverse action notices, accompanied by explicit governance caveats (single-snapshot data, in-sample scoring limitations).
*   **Phase 7 — Responsible-Lending / Fairness Check:** Demographic parity audit across age bands, gender, and education to flag potential disparate impact for independent compliance review — no automatic threshold adjustment is applied or recommended based on protected characteristics.
*   **Phase 8 — Deployment:** FastAPI containerized decisioning endpoint delivering real-time PD scoring, expected loss computation, and top risk driver extraction via SHAP.

---

## Tech Stack

*   **Language:** Python 3.10+
*   **Data & Modeling:** Pandas, NumPy, Scikit-Learn, LightGBM, Statsmodels, Scipy, SHAP
*   **Database & ETL:** SQLAlchemy, SQLite, DBeaver
*   **Deployment:** FastAPI, Uvicorn, Docker, Pydantic

*(Tech stack lists SQLite, matching what the code actually runs. The original draft also listed PostgreSQL/AWS RDS — add that back only if you actually point `db_url` at a real Postgres/RDS instance.)*

---

## Quickstart & Installation

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/nmakoui/lendscope.git
   cd lendscope
   pip install -r requirements.txt
   ```

2. **Fetch the dataset:**
   ```bash
   python fetch_data.py
   ```

3. **Run the pipeline in order:**
   ```bash
   python phase1_etl.py
   python phase2_eda.py
   python phase3_pd_model.py
   python phase4_el_model.py
   python phase5_policy_sim.py
   python phase6_explainability.py
   python phase7_fairness.py
   ```

4. **Run the API:**
   ```bash
   uvicorn app:app --reload
   ```
