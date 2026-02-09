import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

from utils.config import FEATURE_DATA_PATH, ML_MODELS_DIR, DL_MODELS_DIR


# -----------------------------
# LOAD FEATURE DATA
# -----------------------------
def load_feature_data():
    df = pd.read_csv(FEATURE_DATA_PATH, index_col="DateTime", parse_dates=True)
    return df


# -----------------------------
# LOAD SAVED MODELS
# -----------------------------
def load_models():
    lgb_model = joblib.load(ML_MODELS_DIR + "lightgbm_model.pkl")
    xgb_model = joblib.load(ML_MODELS_DIR + "xgboost_model.pkl")
    
    # Define custom objects for legacy metrics
    custom_objects = {
        'mse': 'mean_squared_error',
        'mae': 'mean_absolute_error',
        'mape': 'mean_absolute_percentage_error',
        'cosine': 'cosine_similarity',
        'cosine_proximity': 'cosine_similarity'
    }
    
    lstm_model = load_model(
        DL_MODELS_DIR + "lstm_model.h5",
        custom_objects=custom_objects
    )
    
    cnn_lstm_model = load_model(
        DL_MODELS_DIR + "cnn_lstm_model.h5",
        custom_objects=custom_objects
    )

    return lgb_model, xgb_model, lstm_model, cnn_lstm_model


# -----------------------------
# CREATE SEQUENCES FOR DL MODELS
# -----------------------------
def create_sequences(X, seq_len=60):
    X_seq = []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
    return np.array(X_seq)


# -----------------------------
# HYBRID PREDICTION PIPELINE
# -----------------------------
def hybrid_pipeline():
    print("Loading data...")
    df = load_feature_data()

    # Separate target and features
    y_true = df['Global_active_power'].values
    X = df.drop(columns=['Global_active_power']).values

    # Trim y to match DL sequence shape
    y_true = y_true[60:]

    print("Loading models...")
    lgb_model, xgb_model, lstm_model, cnn_lstm_model = load_models()

    print("Creating sequences for DL...")
    X_seq = create_sequences(X, seq_len=60)

    print("Generating ML Predictions...")
    lgb_pred = lgb_model.predict(X[60:])
    xgb_pred = xgb_model.predict(X[60:])

    print("Generating LSTM Predictions...")
    lstm_pred = lstm_model.predict(X_seq).flatten()

    print("Generating CNN-LSTM Predictions...")
    cnn_lstm_pred = cnn_lstm_model.predict(X_seq).flatten()

    # Weighted Hybrid Prediction
    hybrid_pred = (
        0.4 * lgb_pred +
        0.3 * xgb_pred +
        0.3 * cnn_lstm_pred
    )

    # Evaluation
    mae = mean_absolute_error(y_true, hybrid_pred)
    rmse = np.sqrt(mean_squared_error(y_true, hybrid_pred))
    r2 = r2_score(y_true, hybrid_pred)

    print("\n------ HYBRID MODEL PERFORMANCE -------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save results
    results = pd.DataFrame({
        "Actual": y_true,
        "LightGBM": lgb_pred,
        "XGBoost": xgb_pred,
        "LSTM": lstm_pred,
        "CNN_LSTM": cnn_lstm_pred,
        "Hybrid": hybrid_pred
    })

    results.to_csv("data/processed/hybrid_results.csv")
    print("\nHybrid results saved to data/processed/hybrid_results.csv")

    print("\nHybrid pipeline completed successfully!")

    return results


if __name__ == "__main__":
    hybrid_pipeline()