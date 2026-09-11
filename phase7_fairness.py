import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings("ignore")

print("Loading data and scoring portfolio...")
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

# Train and Score
gbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
gbm.fit(X, y)
df['PD'] = gbm.predict_proba(X)[:, 1]

# 1. Define Policy Threshold
threshold = 0.20
df['Approved'] = df['PD'] < threshold

# Bin Age into readable bands
df['AGE_BAND'] = pd.cut(df['AGE'], bins=[20, 30, 40, 50, 100], labels=['20-30', '31-40', '41-50', '51+'])

print("\n" + "="*60)
print("PHASE 7: FAIRNESS & RESPONSIBLE LENDING CHECK")
print(f"Policy Threshold: Approve if PD < {threshold}")
print("="*60)

# 2. Check Disparities across Demographics
demographics = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE_BAND']

for demo in demographics:
    print(f"\n--- Disparities by {demo} ---")
    summary = df.groupby(demo).apply(
        lambda x: pd.Series({
            'Total_Applicants': len(x),
            'Approval_Rate': x['Approved'].mean(),
            'Loss_Rate_if_Approved': x.loc[x['Approved'], target_col].mean()
        })
    ).reset_index()

    # Format for display
    summary['Total_Applicants'] = summary['Total_Applicants'].astype(int)
    summary['Approval_Rate'] = (summary['Approval_Rate'] * 100).round(1).astype(str) + '%'
    summary['Loss_Rate_if_Approved'] = (summary['Loss_Rate_if_Approved'] * 100).round(1).astype(str) + '%'
    print(summary.to_string(index=False))

# 3. Write-up Output
print("\n" + "="*70)
print("PHASE 7 WRITE-UP: FAIRNESS FINDINGS (COPY-PASTE FOR YOUR REPORT)")
print("="*70)
print("FAIRNESS OBSERVATIONS:")
print("When applying a uniform risk threshold, approval-rate disparities often")
print("emerge across demographic groups due to historical correlations. For")
print("instance, younger applicants may have inherently shorter credit histories,")
print("yielding marginally lower approval rates under a strict model cutoff.")
print("\nRECOMMENDED NEXT STEP (FLAG, NOT FIX):")
print("A detected disparity in approval rate is not, by itself, evidence the")
print("model should be changed - it needs to be investigated before any action is")
print("taken. The correct next step is to flag the disparity for independent")
print("compliance/legal review to determine whether it is explained by a")
print("legitimate, non-discriminatory risk factor, or whether a feature in the")
print("model is acting as a proxy for a protected characteristic (e.g. age,")
print("marital status). Applying a different approval threshold by demographic")
print("group is NOT proposed here: adjusting lending decisions based on")
print("protected characteristics is itself a form of discrimination (disparate")
print("treatment) and is not an appropriate automatic remedy for a disparate-")
print("impact finding. Any policy change should follow from that review, not")
print("precede it.")
print("="*70)
