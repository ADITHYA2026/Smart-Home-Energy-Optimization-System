import pathlib

# Root directory of the project
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Absolute paths (IMPORTANT)
RAW_DATA_PATH = ROOT / "data/raw/household_power_consumption.txt"
CLEAN_DATA_PATH = ROOT / "data/processed/cleaned.csv"
FEATURE_DATA_PATH = ROOT / "data/processed/features.csv"

ML_MODELS_DIR = ROOT / "models/"
DL_MODELS_DIR = ROOT / "models/"