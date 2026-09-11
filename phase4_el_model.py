import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings("ignore")

print("Loading data and retraining LGBM (Phase 3 recap)...")
engine = create_engine('sqlite:///credit_risk.db')
dim = pd.read_sql("SELECT * FROM dim_customer", engine)
fact = pd.read_sql("SELECT * FROM fact_repayment", engine)

# Reshape data
fact_wide = fact.pivot(index='customer_id', columns='month', values=['repayment_status', 'utilisation_ratio', 'payment_ratio'])
fact_wide.columns = [f"{col[0]}_{col[1]}" for col in fact_wide.columns]
fact_wide.reset_index(inplace=True)

df = pd.merge(dim, fact_wide, on='customer_id')

target_col = 'default.payment.next.month'
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
for i in range(1, 7):
    features.extend([f'repayment_status_{i}', f'utilisation_ratio_{i}', f'payment_ratio_{i}'])

X = df[features].fillna(0)
y = df[target_col]

# Retrain LightGBM on the full dataset for portfolio scoring
gbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
gbm.fit(X, y)

print("\n" + "="*50)
print("PHASE 4: EXPECTED LOSS & STRESS TESTING")
print("="*50)

# 1. Base Expected Loss Calculation
print("\nCalculating Base Expected Loss (EL)...")
df['PD_base'] = gbm.predict_proba(X)[:, 1]

# ASSUMPTION: 85% LGD is a standard industry assumption for unsecured revolving credit.
LGD = 0.85
df['EAD'] = df['LIMIT_BAL']

# EL = PD * LGD * EAD
df['EL_base'] = df['PD_base'] * LGD * df['EAD']

portfolio_loss_base = df['EL_base'].sum()
total_exposure = df['EAD'].sum()
# NOTE: LIMIT_BAL in this dataset is denominated in New Taiwan Dollars (NT$),
# not USD - every monetary figure below is NT$.
print(f"Total Portfolio Exposure: NT${total_exposure:,.2f}")
print(f"Base Expected Loss:       NT${portfolio_loss_base:,.2f} ({(portfolio_loss_base/total_exposure)*100:.2f}%)")

# 2. Stress Testing (Downturn Simulation)
print("\nSimulating Downturn Scenario (+1 Delinquency Tier)...")
X_stress = X.copy()

# Shift all repayment status columns by +1 (making everyone 1 month further behind)
for i in range(1, 7):
    X_stress[f'repayment_status_{i}'] = X_stress[f'repayment_status_{i}'] + 1

# Re-predict PD with the stressed features
df['PD_stress'] = gbm.predict_proba(X_stress)[:, 1]
df['EL_stress'] = df['PD_stress'] * LGD * df['EAD']

portfolio_loss_stress = df['EL_stress'].sum()

print(f"Stress Expected Loss:     NT${portfolio_loss_stress:,.2f} ({(portfolio_loss_stress/total_exposure)*100:.2f}%)")
print(f"Total Loss Increase:      +NT${(portfolio_loss_stress - portfolio_loss_base):,.2f}")

print("\n" + "-"*70)
print("METHODOLOGY NOTE:")
print("This model is trained and scored on the same full population (the")
print("current live book), consistent with how a production model would be")
print("scored operationally. It is NOT a held-out, walk-forward evaluation of")
print("EL accuracy - Phase 3's train/test AUC split is the only true")
print("out-of-sample generalisation check in this project. Treat the EL and")
print("stress figures below as illustrative portfolio-level estimates, not a")
print("validated forecast.")
print("-"*70)

print("\nPhase 4 Complete!")
