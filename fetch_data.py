from ucimlrepo import fetch_ucirepo
import pandas as pd

data = fetch_ucirepo(id=350)
X = data.data.features
y = data.data.targets

df = pd.concat([X, y], axis=1)
print(df.shape)
print(df.head())

df.to_csv('credit_default_raw.csv', index=False)
print('Saved to credit_default_raw.csv')
