# Sends one row to the model served with `mlflow models serve` on port 5001

import requests
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
row = data.data.head(1)

payload = {"dataframe_split": row.to_dict(orient="split")}
response = requests.post("http://localhost:5001/invocations", json=payload)
print(response.json())
print("actual:", data.target.iloc[0])
