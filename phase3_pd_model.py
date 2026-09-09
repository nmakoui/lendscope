import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import warnings

# Suppress lightgbm warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Load and Prep Data
print("Loading data from SQLite...")
engine = create_engine('sqlite:///credit_risk.db')
dim = pd.read_sql("SELECT * FROM dim_customer", engine)
fact = pd.read_sql("SELECT * FROM fact_repayment", engine)

# Pivot fact table to wide format for ML models
print("Reshaping data for modeling...")
fact_wide = fact.pivot(index='customer_id', columns='month', values=['repayment_status', 'utilisation_ratio', 'payment_ratio'])
# Flatten the multi-level columns (e.g., repayment_status_1, utilisation_ratio_2)
fact_wide.columns = [f"{col[0]}_{col[1]}" for col in fact_wide.columns]
fact_wide.reset_index(inplace=True)

# Merge back into one big analytical dataset
df = pd.merge(dim, fact_wide, on='customer_id')

# 2. Define Features and Target
target_col = 'default.payment.next.month'
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']

# Add the 6-month historical features dynamically
for i in range(1, 7):
    features.extend([f'repayment_status_{i}', f'utilisation_ratio_{i}', f'payment_ratio_{i}'])

X = df[features].fillna(0)
y = df[target_col]

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n" + "="*50)
print("PHASE 3: PROBABILITY OF DEFAULT (PD) MODEL")
print("="*50)

# 3. Statsmodels Logistic Regression Baseline
print("\nTraining Statsmodels Logistic Regression (Baseline)...")
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit model (using bfgs to help with convergence on this specific dataset)
logit_model = sm.Logit(y_train, X_train_sm).fit(method='bfgs', disp=0)
y_pred_prob_lr = logit_model.predict(X_test_sm)
auc_lr = roc_auc_score(y_test, y_pred_prob_lr)
print(f"Logistic Regression AUC: {auc_lr:.4f}")

# 4. LightGBM Model
print("\nTraining LightGBM Model...")
gbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
gbm.fit(X_train, y_train)

y_pred_prob_lgb = gbm.predict_proba(X_test)[:, 1]
auc_lgb = roc_auc_score(y_test, y_pred_prob_lgb)
print(f"LightGBM AUC:      {auc_lgb:.4f}")

# 5. Feature Importance
print("\nExtracting Top 5 Feature Importances from LightGBM...")
importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance (Gain)': gbm.booster_.feature_importance(importance_type='gain')
}).sort_values(by='Importance (Gain)', ascending=False)

print("-" * 40)
print(importance_df.head(5).to_string(index=False))
print("-" * 40)
print("\nNotice how recent repayment status (repayment_status_1, aka PAY_0) completely dominates, exactly as expected!")