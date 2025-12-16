# src/data.py
import os
import pandas as pd


def load_raw_data(data_dir="data"):
    """
    Train features, labels ve test features'ı yükler.
    Notebook nereden çalışırsa çalışsın dosyaları bulsun diye
    proje köküne göre absolute path kullanır.
    """
    # Bu dosyanın bulunduğu yer: .../proje/src/data.py
    # Proje kökü: .../proje/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, data_dir)

    train_features = pd.read_csv(os.path.join(data_path, "dengue_features_train.csv"))
    train_labels  = pd.read_csv(os.path.join(data_path, "dengue_labels_train.csv"))
    test_features = pd.read_csv(os.path.join(data_path, "dengue_features_test.csv"))

    return train_features, train_labels, test_features


def build_train_dataframe(train_features, train_labels):
    """
    Feature + label birleştirir
    """
    df = train_features.merge(
        train_labels,
        on=["city", "year", "weekofyear"],
        how="left"
    )
    return df


def split_and_clean_by_city(df):
    """
    Şehir bazında ayırır, zaman sırasına dizer, fill işlemlerini yapar
    """
    df_sj = (
        df[df["city"] == "sj"]
        .sort_values(["year", "weekofyear"])
        .reset_index(drop=True)
        .ffill()
        .bfill()
    )

    df_iq = (
        df[df["city"] == "iq"]
        .sort_values(["year", "weekofyear"])
        .reset_index(drop=True)
        .ffill()
        .bfill()
    )

    return df_sj, df_iq
