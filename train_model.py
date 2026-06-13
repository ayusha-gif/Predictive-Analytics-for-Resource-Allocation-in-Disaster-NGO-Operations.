from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "odisha_flood_resource_demand.csv"
MODEL_PATH = ROOT / "models" / "ration_demand_model.joblib"
REPORT_DIR = ROOT / "reports"
TARGET = "dry_ration_kits_needed"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["block", "date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    for lag in [1, 3, 7, 14]:
        df[f"demand_lag_{lag}"] = df.groupby("block")[TARGET].shift(lag)
        df[f"rainfall_lag_{lag}"] = df.groupby("block")["rainfall_mm"].shift(lag)

    df["rainfall_7d_sum"] = (
        df.groupby("block")["rainfall_mm"].shift(1).rolling(7).sum().reset_index(level=0, drop=True)
    )
    df["demand_7d_avg"] = (
        df.groupby("block")[TARGET].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    )
    df["river_3d_avg"] = (
        df.groupby("block")["river_level_m"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    )

    block_dummies = pd.get_dummies(df["block"], prefix="block", dtype=int)
    df = pd.concat([df, block_dummies], axis=1)
    return df.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "rainfall_mm",
        "river_level_m",
        "flood_incidents",
        "population_exposed",
        "road_access_score",
        "month",
        "day_of_year",
        "is_monsoon",
        "demand_lag_1",
        "demand_lag_3",
        "demand_lag_7",
        "demand_lag_14",
        "rainfall_lag_1",
        "rainfall_lag_3",
        "rainfall_lag_7",
        "rainfall_lag_14",
        "rainfall_7d_sum",
        "demand_7d_avg",
        "river_3d_avg",
    ]
    return base + [col for col in df.columns if col.startswith("block_")]


def fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 2.0) -> dict:
    means = X.mean()
    stds = X.std().replace(0, 1)
    X_scaled = (X - means) / stds
    design = np.c_[np.ones(len(X_scaled)), X_scaled.to_numpy()]
    target = y.to_numpy()

    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {"weights": weights, "means": means, "stds": stds}


def predict_ridge(model: dict, X: pd.DataFrame) -> np.ndarray:
    X_scaled = (X - model["means"]) / model["stds"]
    design = np.c_[np.ones(len(X_scaled)), X_scaled.to_numpy()]
    return np.maximum(0, design @ model["weights"])


def build_future_forecast(raw_df: pd.DataFrame, model: dict, features: list[str]) -> pd.DataFrame:
    history = raw_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    future_rows = []
    rng = np.random.default_rng(7)

    for block, block_history in history.groupby("block"):
        block_history = block_history.sort_values("date").copy()
        static = block_history.iloc[-1][["district", "block", "population_exposed", "road_access_score"]].to_dict()

        for step in range(1, 31):
            date = block_history["date"].max() + pd.Timedelta(days=1)
            month = date.month
            is_monsoon = int(month in [6, 7, 8, 9])
            recent_rain = block_history["rainfall_mm"].tail(14).mean()
            rainfall_mm = max(0, recent_rain * 0.65 + rng.gamma(2, 12) * is_monsoon + rng.gamma(1.1, 2.5) * (1 - is_monsoon))
            river_level_m = max(1.2, block_history["river_level_m"].tail(7).mean() * 0.75 + rainfall_mm * 0.02 + rng.normal(0, 0.12))
            flood_incidents = int(rng.poisson(max(0, (rainfall_mm - 60) / 40) + max(0, river_level_m - 4.2)))

            new_row = {
                "date": date,
                **static,
                "rainfall_mm": round(rainfall_mm, 1),
                "river_level_m": round(river_level_m, 2),
                "flood_incidents": flood_incidents,
                TARGET: np.nan,
            }

            temp = pd.concat([block_history, pd.DataFrame([new_row])], ignore_index=True)
            featured = add_features(temp).tail(1)
            X_future = featured.reindex(columns=features, fill_value=0)
            prediction = max(0, int(round(predict_ridge(model, X_future)[0])))
            lower = max(0, int(round(prediction * 0.78)))
            upper = int(round(prediction * 1.22))

            new_row[TARGET] = prediction
            new_row["forecast_lower"] = lower
            new_row["forecast_upper"] = upper
            future_rows.append(new_row)
            block_history = pd.concat([block_history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(future_rows)


def main() -> None:
    raw_df = pd.read_csv(DATA_PATH)
    df = add_features(raw_df)
    features = feature_columns(df)

    cutoff = df["date"].quantile(0.8)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    model = fit_ridge(train[features], train[TARGET])
    predictions = predict_ridge(model, test[features])

    mae = np.mean(np.abs(test[TARGET] - predictions))
    rmse = np.sqrt(np.mean((test[TARGET] - predictions) ** 2))
    mape = np.mean(np.abs((test[TARGET] - predictions) / np.maximum(test[TARGET], 1))) * 100

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump({"model": model, "features": features}, file)

    pd.DataFrame(
        [{"metric": "MAE", "value": mae}, {"metric": "RMSE", "value": rmse}, {"metric": "MAPE", "value": mape}]
    ).to_csv(REPORT_DIR / "accuracy_report.csv", index=False)

    importance = np.abs(model["weights"][1:]) * train[features].std().replace(0, 1).to_numpy()
    pd.DataFrame({"feature": features, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    forecast = build_future_forecast(raw_df, model, features)
    forecast.to_csv(REPORT_DIR / "forecast_next_30_days.csv", index=False)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
