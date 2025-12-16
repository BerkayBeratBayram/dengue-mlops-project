# src/models.py
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


def train_val_split_time(df_city, split_ratio=0.8):
    """
    Zaman sıralı holdout split.
    """
    n = len(df_city)
    cut = int(n * split_ratio)
    train_df = df_city.iloc[:cut].copy()
    val_df = df_city.iloc[cut:].copy()
    return train_df, val_df


def train_eval_xgb_log(df_city, feat_cols, target_col="total_cases", split_ratio=0.8, params=None):
    """
    log1p(target) ile XGB eğitip, val tarafında expm1 ile geri çevirip MAE ölçer.
    """
    if params is None:
        params = {}

    df_city = df_city.sort_values(["year", "weekofyear"]).reset_index(drop=True)
    tr, va = train_val_split_time(df_city, split_ratio=split_ratio)

    X_tr = tr[feat_cols]
    y_tr = np.log1p(tr[target_col].astype(float))

    X_va = va[feat_cols]
    y_va_true = va[target_col].astype(float)

    model = XGBRegressor(
        n_estimators=params.get("n_estimators", 2000),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 4),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        min_child_weight=params.get("min_child_weight", 1.0),
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, np.log1p(y_va_true))],
        verbose=False,
    )

    # val predict -> geri çevir
    pred_log = model.predict(X_va)
    pred = np.expm1(pred_log)

    # negatifleri kırp
    pred = np.clip(pred, 0, None)

    mae = mean_absolute_error(y_va_true, pred)
    return model, mae
