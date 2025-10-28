import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import datetime

# -------------------- Utility Functions -------------------- #

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    return df

def create_dataset(df, features, target, seq_length=60):
    data = df[features].values
    target_data = df[target].values
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(target_data[i])
    return np.array(X), np.array(y)

def direction_accuracy(y_true, y_pred):
    correct = 0
    for i in range(1, len(y_true)):
        true_dir = np.sign(y_true[i] - y_true[i-1])
        pred_dir = np.sign(y_pred[i] - y_pred[i-1])
        if true_dir == pred_dir:
            correct += 1
    return correct / (len(y_true)-1)

# -------------------- Model Function -------------------- #

def build_lstm(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# -------------------- Main Training Function -------------------- #

def run_forex_prediction(csv_path):
    print("\n📊 Loading and preprocessing data...")
    df = load_data(csv_path)

    # Select multiple features
    features = ['Open', 'High', 'Low', 'Close']
    target = 'Close'

    # Scale data
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    seq_length = 100
    X, y = create_dataset(df, features, target, seq_length)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    print(model.summary())

    print("\n🚀 Training model...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, shuffle=False)

    print("\n🔍 Predicting...")
    y_pred = model.predict(X_test)

    # Inverse transform
    dummy = np.zeros((len(y_pred), len(features)))
    dummy[:, features.index(target)] = y_pred.flatten()
    y_pred_real = scaler.inverse_transform(dummy)[:, features.index(target)]

    dummy2 = np.zeros((len(y_test), len(features)))
    dummy2[:, features.index(target)] = y_test.flatten()
    y_test_real = scaler.inverse_transform(dummy2)[:, features.index(target)]

    # Metrics
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    acc = direction_accuracy(y_test_real, y_pred_real)

    print("\n📈 RESULTS:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Directional Accuracy: {acc*100:.2f}%")

    # -------------------- Graphs -------------------- #
    plt.figure(figsize=(12,6))
    plt.plot(y_test_real, label="Actual", color='blue')
    plt.plot(y_pred_real, label="Predicted", color='red')
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss During Training")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    errors = y_test_real - y_pred_real
    plt.hist(errors, bins=30, color='gray', edgecolor='black')
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(y_test_real, y_pred_real, alpha=0.5)
    plt.title("Actual vs Predicted (Scatter)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(errors, color='purple')
    plt.title("Prediction Residuals Over Time")
    plt.xlabel("Samples")
    plt.ylabel("Error")
    plt.show()

    # -------------------- Report -------------------- #
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
    LSTM Forex Price Prediction Report
    -----------------------------------
    Date/Time: {timestamp}

    Dataset: {csv_path}
    Sequence Length: {seq_length}
    Features Used: {features}

    Performance Metrics:
    ---------------------
    MAE  : {mae:.4f}
    RMSE : {rmse:.4f}
    R²   : {r2:.4f}
    Directional Accuracy: {acc*100:.2f} %

    Observations:
    --------------
    - Lower MAE/RMSE means better numeric accuracy.
    - Higher Directional Accuracy means trend prediction is better.
    - Check training vs validation loss for overfitting signs.
    - Try tuning LSTM layers, sequence length, and learning rate to improve results.
    """

    with open("LSTM_Report.txt", "w") as f:
        f.write(report)

    print("\n📝 Report saved as: LSTM_Report.txt")

# -------------------- Run -------------------- #
if __name__ == "__main__":
    run_forex_prediction("datasets/forex_data.csv")
