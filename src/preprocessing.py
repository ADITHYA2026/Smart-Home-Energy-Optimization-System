import pandas as pd
import numpy as np
from utils.config import RAW_DATA_PATH, CLEAN_DATA_PATH


def load_raw_data():
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep=';',
        low_memory=False
    )
    return df


def clean_dataframe(df):
    # Replace ? with NaN
    df.replace('?', np.nan, inplace=True)

    # Convert numerical columns
    for col in df.columns:
        if col not in ['Date', 'Time']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Combine date + time
    df['DateTime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%d/%m/%Y %H:%M:%S")

    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index("DateTime", inplace=True)

    # Interpolate missing values
    df = df.interpolate(method='time')

    # Downsample - 5 minute interval
    df = df.resample("5T").mean()

    return df


def save_clean_data(df):
    df.to_csv(CLEAN_DATA_PATH)
    print(f"Cleaned data saved at: {CLEAN_DATA_PATH}")


def preprocess_pipeline():
    print("Loading dataset...")
    df = load_raw_data()

    print("Cleaning dataset...")
    df = clean_dataframe(df)

    print("Saving cleaned dataset...")
    save_clean_data(df)

    return df


if __name__ == "__main__":
    preprocess_pipeline()