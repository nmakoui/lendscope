import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings("ignore")

print("Loading data and prepping model (Phase 3 recap)...")
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

# Retrain model to get PDs
gbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
gbm.fit(X, y)
df['PD'] = gbm.predict_proba(X)[:, 1]

# Segment by Credit Limit Tier
df['Limit_Tier'] = np.where(df['LIMIT_BAL'] < 100000, 'Low Limit (<100k)', 'High Limit (>=100k)')

print("Simulating Credit Policies...")
thresholds = np.linspace(0.05, 0.80, 100)

# Generate Plot
plt.figure(figsize=(10, 6))

# 1. Overall Portfolio Curve
overall_results = []
for thresh in thresholds:
    approved = df['PD'] < thresh
    app_rate = approved.mean()
    loss_rate = y[approved].mean() if approved.sum() > 0 else np.nan
    overall_results.append({'Approval_Rate': app_rate, 'Loss_Rate': loss_rate})

overall_df = pd.DataFrame(overall_results).dropna()
plt.plot(overall_df['Approval_Rate'], overall_df['Loss_Rate'], label='Overall Portfolio', linewidth=3, color='black')

# 2. Segmented Curves
colors = {'Low Limit (<100k)': '#1f77b4', 'High Limit (>=100k)': '#d62728'}

for tier in df['Limit_Tier'].unique():
    tier_df = df[df['Limit_Tier'] == tier]
    tier_results = []
    
    for thresh in thresholds:
        approved = tier_df['PD'] < thresh
        app_rate = approved.mean()
        loss_rate = tier_df.loc[approved, target_col].mean() if approved.sum() > 0 else np.nan
        tier_results.append({'Approval_Rate': app_rate, 'Loss_Rate': loss_rate})
        
    tr_df = pd.DataFrame(tier_results).dropna()
    plt.plot(tr_df['Approval_Rate'], tr_df['Loss_Rate'], label=tier, linestyle='--', color=colors[tier], linewidth=2)

print("Drawing trade-off curve...")
plt.title('Credit Policy Trade-off: Approval Rate vs. Loss Rate', fontsize=14)
plt.xlabel('Approval Rate (Volume)', fontsize=12)
plt.ylabel('Loss Rate (Actual Default Rate of Approved)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

# Save the plot
image_name = 'policy_tradeoff.png'
plt.savefig(image_name, dpi=300)
print(f"Phase 5 Complete! Plot saved as '{image_name}' in your lendscope folder.")