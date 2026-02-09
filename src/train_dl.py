import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.config import FEATURE_DATA_PATH, DL_MODELS_DIR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam


# -----------------------------
# LOAD FEATURE DATA
# -----------------------------
def load_feature_data():
    df = pd.read_csv(FEATURE_DATA_PATH, index_col='DateTime', parse_dates=True)
    return df


# -----------------------------
# CREATE TIME SERIES SEQUENCES
# -----------------------------
def create_sequences(X, y, seq_len=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


# -----------------------------
# BUILD LSTM MODEL
# -----------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model


# -----------------------------
# BUILD CNN-LSTM MODEL
# -----------------------------
def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model


# -----------------------------
# TRAINING PIPELINE
# -----------------------------
def dl_training_pipeline():
    print("Loading feature dataset...")
    df = load_feature_data()

    # Target and feature separation
    y = df['Global_active_power'].values
    X = df.drop(columns=['Global_active_power']).values

    # Scale features for DL
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("Creating time-series sequences...")
    X_seq, y_seq = create_sequences(X_scaled, y, seq_len=60)

    # Train-test split without shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.20, shuffle=False
    )

    print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])

    # -----------------------------
    # Train LSTM
    # -----------------------------
    print("\nTraining LSTM model...")
    lstm_model = build_lstm_model(input_shape)

    lstm_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    lstm_model.save(DL_MODELS_DIR + "lstm_model.h5")
    print("LSTM model saved!")

    # -----------------------------
    # Train CNN-LSTM
    # -----------------------------
    print("\nTraining CNN-LSTM model...")
    cnn_lstm_model = build_cnn_lstm_model(input_shape)

    cnn_lstm_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    cnn_lstm_model.save(DL_MODELS_DIR + "cnn_lstm_model.h5")
    print("CNN-LSTM model saved!")

    print("\nDeep learning training completed!")


if __name__ == "__main__":
    dl_training_pipeline()