import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy import stats

print("Connecting to SQLite and fetching joined data...")
engine = create_engine('sqlite:///credit_risk.db')

# Pull dim_customer and join with the most recent month (month 1) from fact_repayment
query = """
SELECT 
    c.customer_id, c.AGE, c.EDUCATION, c.MARRIAGE, c."default.payment.next.month" AS default_flag,
    f.repayment_status AS pay_0, f.utilisation_ratio
FROM dim_customer c
JOIN fact_repayment f ON c.customer_id = f.customer_id
WHERE f.month = 1
"""
df = pd.read_sql(query, engine)

def bootstrap_diff(group1, group2, n_iterations=1000):
    """Calculates bootstrap confidence intervals for the difference in means."""
    diffs = []
    # Convert to numpy arrays for much faster sampling
    v1, v2 = group1.values, group2.values 
    for _ in range(n_iterations):
        sample1 = np.random.choice(v1, size=len(v1), replace=True)
        sample2 = np.random.choice(v2, size=len(v2), replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))
    return np.percentile(diffs, [2.5, 97.5]), np.mean(diffs)

# Features we want to hypothesis-test
features = ['pay_0', 'utilisation_ratio', 'AGE', 'EDUCATION', 'MARRIAGE']

defaulters = df[df['default_flag'] == 1]
non_defaulters = df[df['default_flag'] == 0]

print("\n" + "="*50)
print("PHASE 2: STATISTICAL ANALYSIS & BOOTSTRAP CIs")
print("="*50)

for feat in features:
    g1 = defaulters[feat].dropna()
    g2 = non_defaulters[feat].dropna()
    
    # Hypothesis Testing (Welch's t-test for unequal variances)
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    
    # Bootstrapping Effect Size (Difference in Means)
    ci, mean_diff = bootstrap_diff(g1, g2)
    
    print(f"\nFeature: {feat.upper()}")
    print(f"  Mean Defaulters:     {g1.mean():.4f}")
    print(f"  Mean Non-Defaulters: {g2.mean():.4f}")
    print(f"  T-test p-value:      {p_val:.4e}")
    print(f"  Effect Size (Diff):  {mean_diff:.4f}")
    print(f"  95% Bootstrap CI:    [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Quick interpretation logic
    if p_val < 0.05:
        print("  -> SIGNIFICANT difference between groups.")
    else:
        print("  -> NO significant difference.")

print("\nPhase 2 Complete! Review the terminal output.")