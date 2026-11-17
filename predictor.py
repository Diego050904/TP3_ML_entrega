"""Predictor de cancelaciones - TP3 Machine Learning (ITBA 72.75).

Uso:
    from predictor import predictor
    predictions = predictor(active_reservations)   # 1 = cancela, 0 = no cancela, mismo orden

    python predictor.py reservas.csv               # guarda reservas_predictions.csv

Requiere cancellation_model.joblib en la misma carpeta.
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("cancellation_model.joblib")

TARGET = "canceled_job"
LEAKAGE_COLUMNS = ["execution_status", "execution_status_date"]
INPUT_COLUMNS = [
    "hours_before_execution", "execution_year", "execution_month", "execution_week",
    "execution_day", "peak_gpu_hours", "offpeak_gpu_hours", "num_primary_processes",
    "num_auxiliary_processes", "num_background_processes", "service_tier", "client_region",
    "client_type", "request_channel", "returning_client", "previous_job_cancellations",
    "successful_previous_jobs", "reserved_gpu_type", "assigned_gpu_type",
    "resource_modifications", "reservation_commitment", "orchestration_agent_id",
    "organization_id", "queue_wait_time", "workload_profile", "estimated_cost_per_hour",
    "reserved_storage_units", "special_resource_requests", "total_processes",
    "total_gpu_hours", "high_value_client",
]
CATEGORICAL_FEATURES = [
    "service_tier", "client_region", "client_type", "request_channel", "reserved_gpu_type",
    "assigned_gpu_type", "reservation_commitment", "workload_profile",
    "orchestration_agent_id", "organization_id",
]
REFERENCE_DATE = pd.Timestamp("2014-01-01")


def to_reservations_frame(reservations):
    if isinstance(reservations, pd.DataFrame):
        return reservations.drop(columns=[TARGET, *LEAKAGE_COLUMNS], errors="ignore")[INPUT_COLUMNS]
    values = np.atleast_2d(np.asarray(reservations, dtype=object))
    if values.shape[1] == len(INPUT_COLUMNS):
        return pd.DataFrame(values, columns=INPUT_COLUMNS).infer_objects()
    raise ValueError(f"Se esperaban {len(INPUT_COLUMNS)} columnas y llegaron {values.shape[1]}")


def safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def engineer_features(reservations):
    features = to_reservations_frame(reservations).copy()
    numeric_columns = [column for column in INPUT_COLUMNS if column not in CATEGORICAL_FEATURES]
    features[numeric_columns] = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    features["num_auxiliary_processes"] = features["num_auxiliary_processes"].fillna(0)

    execution_date = pd.to_datetime(
        dict(year=features["execution_year"], month=features["execution_month"], day=features["execution_day"]),
        errors="coerce",
    )
    booking_date = execution_date - pd.to_timedelta(features["hours_before_execution"], unit="D")
    features["execution_weekday"] = execution_date.dt.weekday
    features["execution_day_of_year"] = execution_date.dt.dayofyear
    features["execution_days_since_reference"] = (execution_date - REFERENCE_DATE).dt.days
    features["booking_days_since_reference"] = (booking_date - REFERENCE_DATE).dt.days
    features["booking_weekday"] = booking_date.dt.weekday
    features["booking_month"] = booking_date.dt.month

    features["gpu_type_changed"] = (features["reserved_gpu_type"] != features["assigned_gpu_type"]).astype(int)
    features["has_agent"] = features["orchestration_agent_id"].notna().astype(int)
    features["has_organization"] = features["organization_id"].notna().astype(int)
    features["peak_hours_ratio"] = safe_divide(features["peak_gpu_hours"], features["total_gpu_hours"])
    features["cost_per_process"] = safe_divide(features["estimated_cost_per_hour"], features["total_processes"])
    features["total_cost"] = features["estimated_cost_per_hour"] * features["total_gpu_hours"]
    features["previous_jobs"] = features["previous_job_cancellations"] + features["successful_previous_jobs"]
    features["previous_cancellation_ratio"] = safe_divide(features["previous_job_cancellations"], features["previous_jobs"])

    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].map(format_category).fillna("missing")
    return features


def format_category(value):
    if pd.isna(value) or str(value).strip().lower() in ("", "nan", "none"):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


def encode_categories(features, categories):
    encoded = features.copy()
    for column, column_categories in categories.items():
        encoded[column] = pd.Categorical(encoded[column], categories=column_categories)
    return encoded


def load_model_bundle(model_path=MODEL_PATH):
    model_bundle = joblib.load(model_path)
    model_bundle["booster"] = lgb.Booster(model_str=model_bundle["booster_model"])
    return model_bundle


def predict_cancellation_probability(reservations, model_bundle):
    features = encode_categories(engineer_features(reservations), model_bundle["categories"])
    return model_bundle["booster"].predict(features[model_bundle["feature_names"]])


def predictor(reservations):
    model_bundle = load_model_bundle()
    probabilities = predict_cancellation_probability(reservations, model_bundle)
    return (probabilities >= model_bundle["threshold"]).astype(int)


if __name__ == "__main__":
    import sys

    input_path = Path(sys.argv[1])
    output_path = input_path.with_name(f"{input_path.stem}_predictions.csv")
    predictions = predictor(pd.read_csv(input_path))
    pd.Series(predictions, name=TARGET).to_csv(output_path, index=False)
    print(f"{len(predictions):,} predicciones guardadas en {output_path}")
