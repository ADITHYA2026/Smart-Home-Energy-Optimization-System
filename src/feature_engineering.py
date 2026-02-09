import pandas as pd
import numpy as np
from utils.config import CLEAN_DATA_PATH, FEATURE_DATA_PATH


def load_clean_data():
    df = pd.read_csv(CLEAN_DATA_PATH, index_col='DateTime', parse_dates=True)
    return df


def add_time_features(df):
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Seasons: Winter=1, Spring=2, Summer=3, Autumn=4
    df['season'] = df['month'] % 12 // 3 + 1

    return df


def add_lag_features(df):
    df['lag_1'] = df['Global_active_power'].shift(1)
    df['lag_2'] = df['Global_active_power'].shift(2)
    df['lag_3'] = df['Global_active_power'].shift(3)
    return df


def add_rolling_features(df):
    df['rolling_1h'] = df['Global_active_power'].rolling(window=12).mean()
    df['rolling_3h'] = df['Global_active_power'].rolling(window=36).mean()
    df['rolling_6h'] = df['Global_active_power'].rolling(window=72).mean()
    return df


def add_appliance_ratios(df):
    df['ratio_kitchen'] = df['Sub_metering_1'] / (df['Global_active_power'] + 0.001)
    df['ratio_laundry'] = df['Sub_metering_2'] / (df['Global_active_power'] + 0.001)
    df['ratio_ac_heating'] = df['Sub_metering_3'] / (df['Global_active_power'] + 0.001)
    return df


def clean_feature_data(df):
    # Remove rows with NaNs caused by lag/rolling
    df.dropna(inplace=True)
    return df


def save_features(df):
    df.to_csv(FEATURE_DATA_PATH)
    print(f"Feature dataset saved at: {FEATURE_DATA_PATH}")


def feature_engineering_pipeline():
    print("Loading cleaned dataset...")
    df = load_clean_data()

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding lag features...")
    df = add_lag_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding appliance ratio features...")
    df = add_appliance_ratios(df)

    print("Removing NaNs created by lag/rolling...")
    df = clean_feature_data(df)

    print("Saving engineered features...")
    save_features(df)

    return df


if __name__ == "__main__":
    feature_engineering_pipeline()