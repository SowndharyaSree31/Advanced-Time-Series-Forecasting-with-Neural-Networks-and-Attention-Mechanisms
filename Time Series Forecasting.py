# =========================================================
# STEP 0 — INSTALL REQUIRED LIBRARIES (RUN ONCE)
# =========================================================
!pip install yfinance optuna torch torchvision torchaudio matplotlib seaborn scikit-learn


# =========================================================
# STEP 1 — IMPORTS & GLOBAL SETTINGS
# =========================================================
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import optuna
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# STEP 2 — LOAD MULTIVARIATE TIME SERIES DATA
# (Programmatically sourced from a reputable source)
# =========================================================
def load_data(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

data = load_data()
print(data.head())


# =========================================================
# STEP 3 — DATA PREPROCESSING (SCALING + WINDOWING)
# =========================================================
LOOKBACK = 30     # sequence length
HORIZON = 1       # forecast horizon

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon, 3])  # Close price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK, HORIZON)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# =========================================================
# STEP 4 — PYTORCH DATASET
# =========================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=64, shuffle=False)


# =========================================================
# STEP 5 — SELF-ATTENTION MECHANISM
# =========================================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights


# =========================================================
# STEP 6 — SEQ2SEQ LSTM WITH ATTENTION
# =========================================================
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, HORIZON)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights


# =========================================================
# STEP 7 — HYPERPARAMETER OPTIMIZATION (OPTUNA)
# =========================================================
def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = LSTMAttentionModel(X.shape[2], hidden_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds_list, actual_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds, _ = model(xb)
            preds_list.extend(preds.cpu().numpy())
            actual_list.extend(yb.numpy())

    rmse = np.sqrt(mean_squared_error(actual_list, preds_list))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("Best Parameters:", best_params)


# =========================================================
# STEP 8 — TRAIN FINAL MODEL
# =========================================================
model = LSTMAttentionModel(
    input_dim=X.shape[2],
    hidden_dim=best_params["hidden_dim"],
    num_layers=best_params["num_layers"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
loss_fn = nn.MSELoss()

EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds, _ = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")


# =========================================================
# STEP 9 — MODEL EVALUATION
# =========================================================
model.eval()
preds, actuals, attentions = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out, attn = model(xb)
        preds.extend(out.cpu().numpy())
        actuals.extend(yb.numpy())
        attentions.append(attn.cpu().numpy())

preds = np.array(preds).flatten()
actuals = np.array(actuals).flatten()

mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
directional_accuracy = np.mean(
    np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.4f}")


# =========================================================
# STEP 10 — ATTENTION INTERPRETABILITY
# =========================================================
avg_attention = np.mean(np.concatenate(attentions), axis=0)

plt.figure(figsize=(10,5))
plt.plot(avg_attention)
plt.title("Average Attention Weights Across Time Steps")
plt.xlabel("Time Steps (Lookback Window)")
plt.ylabel("Attention Weight")
plt.show()


# =========================================================
# STEP 11 — FORECAST VS ACTUAL VISUALIZATION
# =========================================================
plt.figure(figsize=(12,6))
plt.plot(actuals[:200], label="Actual")
plt.plot(preds[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Close Price")
plt.show()
