import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from utils.config import FEATURE_DATA_PATH, ML_MODELS_DIR


def load_feature_data():
    df = pd.read_csv(FEATURE_DATA_PATH, index_col='DateTime', parse_dates=True)
    return df


def train_test_split_data(df):
    X = df.drop(columns=['Global_active_power'])
    y = df['Global_active_power']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return mae, rmse, r2


def train_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    path = ML_MODELS_DIR + filename
    joblib.dump(model, path)
    print(f"Model saved at {path}")


def ml_training_pipeline():
    print("Loading feature dataset...")
    df = load_feature_data()

    print("Splitting train-test data...")
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train)
    mae, rmse, r2 = evaluate(lgb_model, X_test, y_test)
    print(f"LightGBM → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    save_model(lgb_model, "lightgbm_model.pkl")

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    mae, rmse, r2 = evaluate(xgb_model, X_test, y_test)
    print(f"XGBoost → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    save_model(xgb_model, "xgboost_model.pkl")

    print("ML training completed!")


if __name__ == "__main__":
    ml_training_pipeline()