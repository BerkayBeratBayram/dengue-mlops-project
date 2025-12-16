# src/train.py
import os
import json
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from src.data import load_raw_data, build_train_dataframe, split_and_clean_by_city
from src.features import build_features_city_concat, get_feature_cols
from src.models import train_eval_xgb_log


def main():
    # 1) yükle
    train_features, train_labels, test_features = load_raw_data()
    df = build_train_dataframe(train_features, train_labels)

    # 2) train şehir ayır
    df_sj, df_iq = split_and_clean_by_city(df)

    # 3) test şehir ayır
    test_sj = (
        test_features[test_features["city"] == "sj"]
        .sort_values(["year", "weekofyear"])
        .reset_index(drop=True)
    )
    test_iq = (
        test_features[test_features["city"] == "iq"]
        .sort_values(["year", "weekofyear"])
        .reset_index(drop=True)
    )

    # 4) feature üret (train+test concat ile uyum garantisi)
    tr_sj, te_sj = build_features_city_concat(df_sj, test_sj, include_cases_lags=False)
    tr_iq, te_iq = build_features_city_concat(df_iq, test_iq, include_cases_lags=False)

    feat_cols_sj = get_feature_cols(tr_sj)
    feat_cols_iq = get_feature_cols(tr_iq)

    # 5) model params (baseline)
    params_sj = dict(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    params_iq = dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    # =======================
    # 🔥 MLFLOW BAŞLANGIÇ
    # =======================
    mlflow.set_experiment("dengue_forecasting")

    with mlflow.start_run(run_name="xgb_log_sj_iq"):
        # params log
        mlflow.log_params({f"sj_{k}": v for k, v in params_sj.items()})
        mlflow.log_params({f"iq_{k}": v for k, v in params_iq.items()})

        # 6) train / eval
        model_sj, mae_sj = train_eval_xgb_log(
            tr_sj, feat_cols_sj, params=params_sj
        )
        model_iq, mae_iq = train_eval_xgb_log(
            tr_iq, feat_cols_iq, params=params_iq
        )

        # metrics log
        mlflow.log_metric("mae_sj", float(mae_sj))
        mlflow.log_metric("mae_iq", float(mae_iq))

        print(f"SJ val MAE: {mae_sj:.4f} | features: {len(feat_cols_sj)}")
        print(f"IQ val MAE: {mae_iq:.4f} | features: {len(feat_cols_iq)}")

        # 7) artifacts (dosya + MLflow)
        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(model_sj, "artifacts/model_sj.pkl")
        joblib.dump(model_iq, "artifacts/model_iq.pkl")

        with open("artifacts/feat_cols_sj.json", "w", encoding="utf-8") as f:
            json.dump(feat_cols_sj, f, ensure_ascii=False, indent=2)

        with open("artifacts/feat_cols_iq.json", "w", encoding="utf-8") as f:
            json.dump(feat_cols_iq, f, ensure_ascii=False, indent=2)

        with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {"mae_sj": mae_sj, "mae_iq": mae_iq},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # MLflow artifacts
        mlflow.log_artifact("artifacts/feat_cols_sj.json")
        mlflow.log_artifact("artifacts/feat_cols_iq.json")
        mlflow.log_artifact("artifacts/metrics.json")

        # MLflow model kayıt
        mlflow.sklearn.log_model(model_sj, artifact_path="model_sj")
        mlflow.sklearn.log_model(model_iq, artifact_path="model_iq")

        print("✅ MLflow run tamamlandı, artifacts kaydedildi.")


if __name__ == "__main__":
    main()
