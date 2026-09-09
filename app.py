from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="LendScope API")

# 1. Train Model on Startup
engine = create_engine('sqlite:///credit_risk.db')
dim = pd.read_sql("SELECT * FROM dim_customer", engine)
fact = pd.read_sql("SELECT * FROM fact_repayment", engine)
fact_wide = fact.pivot(index='customer_id', columns='month', values=['repayment_status', 'utilisation_ratio', 'payment_ratio'])
fact_wide.columns = [f"{col[0]}_{col[1]}" for col in fact_wide.columns]
fact_wide.reset_index(inplace=True)
df = pd.merge(dim, fact_wide, on='customer_id')

features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
for i in range(1, 7):
    features.extend([f'repayment_status_{i}', f'utilisation_ratio_{i}', f'payment_ratio_{i}'])

X = df[features].fillna(0)
y = df['default.payment.next.month']

gbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
gbm.fit(X, y)
explainer = shap.TreeExplainer(gbm)

# 2. Define Input Data Schema
class CustomerData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    repayment_status_1: int
    repayment_status_2: int
    repayment_status_3: int
    repayment_status_4: int
    repayment_status_5: int
    repayment_status_6: int
    utilisation_ratio_1: float
    utilisation_ratio_2: float
    utilisation_ratio_3: float
    utilisation_ratio_4: float
    utilisation_ratio_5: float
    utilisation_ratio_6: float
    payment_ratio_1: float
    payment_ratio_2: float
    payment_ratio_3: float
    payment_ratio_4: float
    payment_ratio_5: float
    payment_ratio_6: float

# 3. Prediction Endpoint
@app.post("/predict")
def predict(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])
    
    # Predict PD
    pd_val = gbm.predict_proba(input_data)[0, 1]
    
    # Expected Loss (LGD = 85%, EAD = LIMIT_BAL)
    el = pd_val * 0.85 * data.LIMIT_BAL
    
    # SHAP Explanation (find the highest contributing feature)
    shap_vals = explainer.shap_values(input_data)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1][0]
    else:
        shap_vals = shap_vals[0]
        
    top_feature_idx = np.argmax(np.abs(shap_vals))
    top_feature = features[top_feature_idx]
    
    return {
        "Probability_of_Default": f"{pd_val*100:.2f}%",
        "Expected_Loss": f"${el:,.2f}",
        "Decision": "Approve" if pd_val < 0.20 else "Decline",
        "Top_Risk_Driver": top_feature,
        "Driver_Impact": "Increases Risk" if shap_vals[top_feature_idx] > 0 else "Decreases Risk"
    }