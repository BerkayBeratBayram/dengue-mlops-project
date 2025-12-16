# src/features.py
import numpy as np
import pandas as pd

ID_COLS = ["city", "year", "weekofyear", "week_start_date"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mevsimsellik: weekofyear -> sin/cos"""
    out = df.copy()
    w = out["weekofyear"].astype(int)
    out["week_sin"] = np.sin(2 * np.pi * w / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * w / 52.0)
    return out


def add_basic_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Basit etkileşimler / türetilmiş feature'lar"""
    out = df.copy()

    # sıcaklık aralığı (varsa)
    if ("reanalysis_max_air_temp_k" in out.columns) and ("reanalysis_min_air_temp_k" in out.columns):
        out["reanalysis_temp_range_k"] = out["reanalysis_max_air_temp_k"] - out["reanalysis_min_air_temp_k"]

    # nem * sıcaklık (varsa)
    if ("reanalysis_relative_humidity_percent" in out.columns) and ("reanalysis_air_temp_k" in out.columns):
        out["temp_humidity_interaction"] = (
            out["reanalysis_air_temp_k"] * out["reanalysis_relative_humidity_percent"]
        )

    return out


def add_rolling_features(df: pd.DataFrame, roll_windows=(3, 5), roll_cols=None) -> pd.DataFrame:
    """
    Rolling mean özellikleri.
    NOT: leakage olmaması için rolling -> geçmişe dönük (shift(1)) ile yapılır.
    """
    out = df.copy()

    # Varsayılan: sık kullanılan meteoroloji kolonları varsa onları seç
    if roll_cols is None:
        candidate = [
            "reanalysis_specific_humidity_g_per_kg",
            "reanalysis_dew_point_temp_k",
            "reanalysis_air_temp_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_max_air_temp_k",
            "station_avg_temp_c",
            "station_min_temp_c",
            "station_max_temp_c",
            "precipitation_amt_mm",
            "reanalysis_precip_amt_kg_per_m2",
        ]
        roll_cols = [c for c in candidate if c in out.columns]

    # Zaman sırası garanti
    out = out.sort_values(["city", "year", "weekofyear"]).reset_index(drop=True)

    g = out.groupby("city", sort=False)

    for c in roll_cols:
        for w in roll_windows:
            out[f"{c}_roll{w}"] = g[c].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    return out


def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3), lag_cols=None) -> pd.DataFrame:
    """
    Lag özellikleri (meteorolojik kolonlar için).
    """
    out = df.copy()

    if lag_cols is None:
        candidate = [
            "reanalysis_specific_humidity_g_per_kg",
            "reanalysis_dew_point_temp_k",
            "reanalysis_air_temp_k",
            "station_avg_temp_c",
            "precipitation_amt_mm",
        ]
        lag_cols = [c for c in candidate if c in out.columns]

    out = out.sort_values(["city", "year", "weekofyear"]).reset_index(drop=True)
    g = out.groupby("city", sort=False)

    for c in lag_cols:
        for lag in lags:
            out[f"{c}_lag{lag}"] = g[c].shift(lag)

    return out


def build_features_city_concat(
    train_city: pd.DataFrame,
    test_city: pd.DataFrame,
    roll_windows=(3, 5),
    lags=(1, 2, 3),
    include_cases_lags=False,
    cases_lags=(1, 2, 3),
    target_col="total_cases",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train + test'i birleştirip feature üretir, sonra tekrar ayırır.
    Böylece train/test kolonları %100 uyumlu olur.

    include_cases_lags=True yaparsan:
    - train tarafında total_cases üzerinden lag üretir
    - test tarafında bu kolonlar NaN kalır (sonra predict aşamasında doldurulur)
    """
    tr = train_city.copy()
    te = test_city.copy()

    # test'te target yoksa ekle (concat için)
    if target_col not in te.columns:
        te[target_col] = np.nan

    tr["_is_train"] = 1
    te["_is_train"] = 0

    all_df = pd.concat([tr, te], axis=0, ignore_index=True)

    # Feature adımları
    all_df = add_time_features(all_df)
    all_df = add_basic_interactions(all_df)
    all_df = add_rolling_features(all_df, roll_windows=roll_windows)
    all_df = add_lag_features(all_df, lags=lags)

    # Opsiyonel: case lag (train'de faydalı ama testte doldurma gerekir)
    if include_cases_lags and (target_col in all_df.columns):
        all_df = all_df.sort_values(["city", "year", "weekofyear"]).reset_index(drop=True)
        g = all_df.groupby("city", sort=False)
        for lag in cases_lags:
            all_df[f"cases_lag_{lag}"] = g[target_col].shift(lag)

    # Eksikleri doldur: sadece feature kolonlarında
    # (ID + target + flag hariç)
    drop_for_fill = set(ID_COLS + [target_col, "_is_train"])
    feat_candidates = [c for c in all_df.columns if c not in drop_for_fill]

    # şehir içinde ffill/bfill (meteorolojik veri için mantıklı)
    all_df = all_df.sort_values(["city", "year", "weekofyear"]).reset_index(drop=True)
    all_df[feat_candidates] = all_df.groupby("city", sort=False)[feat_candidates].ffill().bfill()

    # tekrar ayır
    tr_done = all_df[all_df["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    te_done = all_df[all_df["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)

    return tr_done, te_done


def get_feature_cols(df: pd.DataFrame, target_col="total_cases", extra_drop_cols=None) -> list[str]:
    """Modele girecek sayısal feature kolonlarını döndürür."""
    if extra_drop_cols is None:
        extra_drop_cols = []

    drop_cols = set(ID_COLS + [target_col, "log_cases"] + list(extra_drop_cols))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in drop_cols]
