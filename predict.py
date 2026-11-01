# Finds the best run (lowest rmse) in the experiment, loads that model
# and predicts on a few rows of the diabetes data.

import mlflow
import mlflow.pyfunc
from sklearn.datasets import load_diabetes

mlflow.set_tracking_uri("sqlite:///mlflow.db")

runs = mlflow.search_runs(
    experiment_names=["diabetes-progression"],
    order_by=["metrics.rmse ASC"],
    max_results=1,
)
best = runs.iloc[0]
print("Best run:", best["tags.mlflow.runName"])
print("Run id:", best["run_id"])
print("RMSE:", round(best["metrics.rmse"], 2))

model = mlflow.pyfunc.load_model(f"runs:/{best['run_id']}/model")

data = load_diabetes(as_frame=True)
sample = data.data.head(5)
preds = model.predict(sample)

for p, actual in zip(preds, data.target.head(5)):
    print(f"predicted {p:.1f}   actual {actual:.1f}")

# MLflow writes a requirements.txt next to every logged model.
# Print the one for the best run so we know what's needed to serve it.
req_path = mlflow.pyfunc.get_model_dependencies(f"runs:/{best['run_id']}/model")
print("\nModel requirements.txt:")
with open(req_path) as f:
    print(f.read())
