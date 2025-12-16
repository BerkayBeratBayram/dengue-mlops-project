# src/predict.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from src.data import load_raw_data, build_train_dataframe, split_and_clean_by_city
from src.features import build_features_city_concat


def main(
    data_dir="data",
    artifacts_dir="artifacts",
    out_path="submission.csv",
):
    # 1) Model + feature listeleri yükle
    model_sj = joblib.load(os.path.join(artifacts_dir, "model_sj.pkl"))
    model_iq = joblib.load(os.path.join(artifacts_dir, "model_iq.pkl"))

    with open(os.path.join(artifacts_dir, "feat_cols_sj.json"), "r", encoding="utf-8") as f:
        feat_cols_sj = json.load(f)
    with open(os.path.join(artifacts_dir, "feat_cols_iq.json"), "r", encoding="utf-8") as f:
        feat_cols_iq = json.load(f)

    # 2) Veriyi yükle
    train_features, train_labels, test_features = load_raw_data(data_dir=data_dir)
    df_train = build_train_dataframe(train_features, train_labels)

    # 3) Train'i şehir bazında ayır
    df_sj, df_iq = split_and_clean_by_city(df_train)

    # 4) Test'i şehir bazında ayır
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

    # 5) Feature üret (train+test concat => kolon uyumu garanti)
    tr_sj, te_sj = build_features_city_concat(df_sj, test_sj, include_cases_lags=False)
    tr_iq, te_iq = build_features_city_concat(df_iq, test_iq, include_cases_lags=False)

    # 6) Tahmin (model log1p ile eğitildiği için expm1 geri çevir)
    pred_sj = np.expm1(model_sj.predict(te_sj[feat_cols_sj]))
    pred_iq = np.expm1(model_iq.predict(te_iq[feat_cols_iq]))

    # negatifleri kırp
    pred_sj = np.clip(pred_sj, 0, None)
    pred_iq = np.clip(pred_iq, 0, None)

    # 7) submission_format.csv yükle ve doldur
    sub_format = pd.read_csv(os.path.join(data_dir, "submission_format.csv"))

    # format dosyası genelde city/year/weekofyear sıralı gelir
    # biz city bazında dolduracağız
    mask_sj = sub_format["city"] == "sj"
    mask_iq = sub_format["city"] == "iq"

    sub_format.loc[mask_sj, "total_cases"] = pred_sj
    sub_format.loc[mask_iq, "total_cases"] = pred_iq

    # Kaggle/DrivenData beklediği gibi int yapmak genelde iyi (ama yuvarlama kararı sende)
    sub_format["total_cases"] = sub_format["total_cases"].round().astype(int)

    # 8) Kaydet
    sub_format.to_csv(out_path, index=False)
    print(f"✅ Submission kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
