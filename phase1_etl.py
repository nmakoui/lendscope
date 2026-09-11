import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# 1. Setup Data & Connection
file_path = "credit_default_raw.csv"  # relative path - run this script from the project folder
db_url = 'sqlite:///credit_risk.db'
engine = create_engine(db_url)

print("Loading raw data...")
# Read normally, no header skipping needed
df = pd.read_csv(file_path)

# 1.5 Rename X1-Y to actual features
column_mapping = {
    'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 'X5': 'AGE',
    'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 'X10': 'PAY_5', 'X11': 'PAY_6',
    'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4', 'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3', 'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6',
    'Y': 'default.payment.next.month'
}
df.rename(columns=column_mapping, inplace=True)

# Ensure ID column exists for joining later
if 'ID' not in df.columns:
    df.insert(0, 'ID', range(1, len(df) + 1))

# 2. Build Customer Dimension Table
dim_cols = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'default.payment.next.month']
dim_customer = df[dim_cols].copy()
dim_customer.rename(columns={'ID': 'customer_id'}, inplace=True)

# 3. Build Repayment Fact Table (Wide to Long)
pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

fact_dfs = []
for i in range(1, 7):
    temp_df = pd.DataFrame({
        'customer_id': df['ID'],
        'month': i,
        'repayment_status': df[pay_status_cols[i-1]],
        'bill_amt': df[f'BILL_AMT{i}'],
        'pay_amt': df[f'PAY_AMT{i}'],
        'limit_bal': df['LIMIT_BAL']
    })
    fact_dfs.append(temp_df)

fact_table = pd.concat(fact_dfs, ignore_index=True)

# 4. Engineer Derived Fields
print("Engineering features...")
fact_table['utilisation_ratio'] = fact_table['bill_amt'] / fact_table['limit_bal']

# Guard against division by zero
fact_table['payment_ratio'] = np.where(
    fact_table['bill_amt'] > 0,
    fact_table['pay_amt'] / fact_table['bill_amt'],
    0
)
fact_table.drop(columns=['limit_bal'], inplace=True)

# 5. Load into SQLite
# NOTE: this project uses SQLite locally (see db_url above). The tech stack
# lists PostgreSQL/AWS RDS as the intended production target for this schema -
# if you want that claim to hold, point db_url at a real Postgres/RDS instance
# instead of leaving this on SQLite.
print("Pushing to SQLite...")
dim_customer.to_sql('dim_customer', engine, if_exists='replace', index=False)
fact_table.to_sql('fact_repayment', engine, if_exists='replace', index=False)

print("Phase 1 Complete! Check DBeaver.")
