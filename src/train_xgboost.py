import os
import sys
from datetime import datetime

import mlflow
import optuna
from dotenv import load_dotenv
from mlflow.tracking.client import MlflowClient
from optuna.integration.mlflow import MLflowCallback
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from helper_functions import get_data, preprocess

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
load_dotenv()


# TODO: make logging work for all runs of optuna


def train():
    """Train the model."""
    dataset = get_data()
    # df_processed = preprocess(dataset)

    y = dataset["trip_duration_minutes"].values
    X = dataset.drop(columns=["trip_duration_minutes"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    features = ["PULocationID", "DOLocationID", "trip_distance"]
    target = "duration"

    load_dotenv()
    year = os.getenv("YEAR")
    month = int(os.getenv("MONTH"))
    color = os.getenv("COLOR")
    tags = {
        "model": "random forest regressor",
        "dataset": f"{color}-taxi",
        "year": year,
        "month": month,
        "features": features,
        "target": target,
    }

    mlflc = MLflowCallback(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI_local"),
        metric_name="RMSE",
        mlflow_kwargs={"tags": tags},
    )

    @mlflc.track_in_mlflow()
    def objective(trial):
        search_space = {
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.00001, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.000001, 0.1, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.1, 1000, log=True
            ),
            "objective": "reg:squarederror",
            "seed": 42,
        }

        xgb_regressor = make_pipeline(
            DictVectorizer(),
            XGBRegressor(
                max_depth=search_space["max_depth"],
                learning_rate=search_space["learning_rate"],
                reg_alpha=search_space["reg_alpha"],
                reg_lambda=search_space["reg_lambda"],
                min_child_weight=search_space["min_child_weight"],
                objective=search_space["objective"],
                random_state=search_space["seed"],
                verbosity=2,
                n_jobs=8,
            ),
        )

        # Train the regressor
        xgb_regressor.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = xgb_regressor.predict(X_test)

        # Calculate the root mean squared error (RMSE)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        return rmse

    # Perform hyperparameter optimization with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, callbacks=[mlflc], n_jobs=8)

    # Get the best trial and parameters
    best_trial = study.best_trial
    best_params = best_trial.params

    # Log the Optuna search results to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("best_rmse", best_trial.value)

    xgb_regressor = make_pipeline(
        DictVectorizer(),
        XGBRegressor(
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            min_child_weight=best_params["min_child_weight"],
            objective=best_params["objective"],
            random_state=best_params["seed"],
            n_jobs=8,
        ),
    )

    print("Training the model...")
    xgb_regressor.fit(X_train, y_train)

    y_pred = xgb_regressor.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(xgb_regressor, "xgboost-fare-model")
    # mlflow.log_artifact("dict_vectorizer.pkl")
    run_id = mlflow.active_run().info.run_id
    print("Model run id")
    print(run_id)
    return run_id


def register_model(RUN_ID):
    # TODO: registering is not working here, you still need to
    #  do it manually from the MLflow UI.
    model_uri = f"runs:/{RUN_ID}/model"

    load_dotenv()
    MLFLOW_TRACKING_URI_local = os.environ.get("MLFLOW_TRACKING_URI_local")
    mlflow.register_model(model_uri=model_uri, name="xgboost-fare-model")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI_local)
    model_name = "xgboost-fare-model"
    latest_versions = client.get_latest_versions(name=model_name, stages=["None"])

    for version in latest_versions:
        print(f"version: {version.version}, stage: {version.current_stage}")
    model_version = latest_versions[-1].version
    new_stage = "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True,
    )
    date = datetime.today().date()
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_stage} on {date}",
    )


if __name__ == "__main__":
    run_id = train()
    register_model(run_id)
