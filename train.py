# MLflow experiment tracking lab
# Dataset: sklearn diabetes (disease progression one year after baseline)
# Trains a few regression models, logs each one as an MLflow run,
# then registers the best one in the Model Registry.

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed, we only save the plot to a file
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# SQLite backend instead of the default mlruns file store,
# so the Model Registry works locally
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes-progression")

REGISTERED_NAME = "DiabetesProgressionModel"


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def save_pred_plot(actual, pred, title, path):
    # predicted vs actual, points on the dashed line = perfect prediction
    plt.figure(figsize=(5, 5))
    plt.scatter(actual, pred, alpha=0.6)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_model(name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, preds)

        print(f"{name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

        mlflow.set_tag("model_type", model.__class__.__name__)
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # extra artifact: predicted vs actual plot
        plot_file = "pred_vs_actual.png"
        save_pred_plot(y_test, preds, name, plot_file)
        mlflow.log_artifact(plot_file)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature,
                                 input_example=X_train.head(2))

        return run.info.run_id, rmse


if __name__ == "__main__":
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # optional command line args: python train.py <ridge_alpha> <n_estimators>
    ridge_alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    n_trees = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    candidates = [
        ("Ridge alpha=" + str(ridge_alpha), Ridge(alpha=ridge_alpha),
         {"alpha": ridge_alpha}),
        ("Lasso alpha=0.1", Lasso(alpha=0.1), {"alpha": 0.1}),
        ("Lasso alpha=1.0", Lasso(alpha=1.0), {"alpha": 1.0}),
    ]
    for depth in [3, 6]:
        candidates.append((
            f"RandomForest n={n_trees} depth={depth}",
            RandomForestRegressor(n_estimators=n_trees, max_depth=depth, random_state=42),
            {"n_estimators": n_trees, "max_depth": depth},
        ))
    for lr in [0.05, 0.1]:
        candidates.append((
            f"GradBoost lr={lr}",
            GradientBoostingRegressor(learning_rate=lr, n_estimators=100, random_state=42),
            {"learning_rate": lr, "n_estimators": 100},
        ))

    results = []
    for name, model, params in candidates:
        run_id, rmse = run_model(name, model, params, X_train, X_test, y_train, y_test)
        results.append((rmse, run_id, name))

    # register the run with the lowest test RMSE
    best_rmse, best_run_id, best_name = min(results)
    print(f"\nBest model: {best_name} (RMSE={best_rmse:.2f}), run id {best_run_id}")
    mlflow.register_model(f"runs:/{best_run_id}/model", REGISTERED_NAME)
