"""Data loading, splitting and the leakage-safe preprocessing step (SRS section 4.2)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from . import config


def load_data(path=config.DATA_PATH, drop_duplicates=True, verbose=True):
    """Load creditcard.csv, report quality, and drop exact duplicate rows.

    Duplicates are removed *before* the split so the same row can never sit
    in both train and test.
    """
    df = pd.read_csv(path)
    df[config.TARGET] = df[config.TARGET].astype(int)
    if verbose:
        print(f"Shape: {df.shape}")
        print(f"Missing values: {int(df.isna().sum().sum())}")
        print(f"Duplicate rows: {int(df.duplicated().sum())}")
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        if verbose:
            print(f"Shape after dropping duplicates: {df.shape}")
    return df


def split_xy(df):
    return df.drop(columns=config.TARGET), df[config.TARGET]


def train_test(df):
    """Stratified 80/20 split with the fixed seed. X_test is locked after this."""
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )


def make_preprocessor():
    """RobustScaler on Time and Amount only; V1-V28 pass through unchanged.

    It is used as the first step of an imblearn Pipeline, so it is fitted on
    training folds only.
    """
    return ColumnTransformer(
        [("robust", RobustScaler(), config.SCALE_COLS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
