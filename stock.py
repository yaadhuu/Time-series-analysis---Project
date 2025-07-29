

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from prophet import Prophet
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


ticker = 'TSLA'
start_date = '2018-01-01'
end_date = '2025-01-01'

data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
data.reset_index(inplace=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [
        '_'.join([str(x) for x in col if x]) if isinstance(col, tuple) else str(col)
        for col in data.columns.values
    ]
else:
    data.columns = data.columns.astype(str)


possible_close_cols = ['Close', 'Adj Close', 'TSLA_Close', 'TSLA_Adj Close', 'Close_TSLA', 'Adj Close_TSLA']
close_col = None
for c in possible_close_cols:
    if c in data.columns:
        close_col = c
        break
if close_col is None:
    raise KeyError(f"Could not find close price column. Available columns: {list(data.columns)}")


df = data[['Date', close_col]].rename(columns={close_col: 'Close'})
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


df['Naive'] = df['Close'].shift(1)  # Yesterday’s price
df['MA_20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average
df['Return'] = df['Close'].pct_change()
df['Volatility_20'] = df['Return'].rolling(window=20).std()
df.dropna(inplace=True)  # Drop initial rows with NA


df_prophet = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

if isinstance(df_prophet['y'], pd.DataFrame):
    df_prophet['y'] = df_prophet['y'].iloc[:, 0]

df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
df_prophet.dropna(subset=['y'], inplace=True)

prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(df_prophet)

future = prophet_model.make_future_dataframe(periods=60)
forecast = prophet_model.predict(future)


scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

train_size = int(len(scaled_close)*0.8)
train_data = scaled_close[:train_size]
test_data = scaled_close[train_size - 60:]

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)

lstm_pred_scaled = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
lstm_dates = df.index[train_size:train_size + len(lstm_pred)]


series = df['Close']

arima_model = ARIMA(series, order=(5,1,0)).fit()
arima_forecast = arima_model.forecast(steps=60)
arima_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=60)

sarima_model = SARIMAX(series, order=(2,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
sarima_forecast = sarima_model.forecast(steps=60)
sarima_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=60)


returns = 100 * df['Close'].pct_change().dropna()
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')
garch_forecast = garch_fit.forecast(horizon=60)
volatility_forecast = np.sqrt(garch_forecast.variance.values[-1, :])


rolling_mean = df['Close'].rolling(window=30).mean()
rolling_std = df['Close'].rolling(window=30).std()
df['Z-Score'] = (df['Close'] - rolling_mean) / rolling_std
df['Anomaly'] = df['Z-Score'].abs() > 2.5
df_anomalies = df[df['Anomaly']]

signals = (lstm_pred > y_true).astype(int)  # Buy signal if predicted price > actual price
returns_for_sim = df.loc[lstm_dates, 'Return'][:len(signals)].values
strategy_returns = signals * returns_for_sim
cumulative_strategy_returns = np.cumprod(1 + strategy_returns) - 1
cumulative_buy_hold_returns = np.cumprod(1 + returns_for_sim) - 1


def evaluate_model(name, actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")

print("\n=== Model Evaluation on Last 60 Days ===")
evaluate_model("LSTM", y_true, lstm_pred)
evaluate_model("ARIMA", series[-60:], arima_forecast.values)
evaluate_model("SARIMA", series[-60:], sarima_forecast.values)
evaluate_model("Prophet", df_prophet['y'].iloc[-60:], forecast['yhat'].iloc[-60:])


prophet_fig = prophet_model.plot(forecast)
plt.title('Prophet Forecast - Tesla')
plt.show()


prophet_compo_fig = prophet_model.plot_components(forecast)
plt.suptitle('Prophet Forecast Components - Tesla')
plt.show()


plt.figure(figsize=(12,6))
plt.plot(df.index[:train_size], df['Close'].iloc[:train_size], label='Train')
plt.plot(lstm_dates, y_true, label='Actual')
plt.plot(lstm_dates, lstm_pred, label='LSTM Prediction')
plt.title('LSTM Forecast - Tesla')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(volatility_forecast, label='Forecasted Volatility')
plt.title('GARCH Volatility Forecast (Next 60 Days)')
plt.xlabel('Days Ahead')
plt.ylabel('Volatility (%)')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(x=df.index, y=df['Close'], label='Close Price')
sns.scatterplot(x=df_anomalies.index, y=df_anomalies['Close'], color='red', label='Anomaly', s=100)
plt.title('Rolling Z-Score Anomaly Detection')
plt.show()


plt.figure(figsize=(12,6))
plt.plot(lstm_dates[:len(cumulative_strategy_returns)], cumulative_strategy_returns, label='LSTM Strategy')
plt.plot(lstm_dates[:len(cumulative_buy_hold_returns)], cumulative_buy_hold_returns, label='Buy & Hold', linestyle='--')
plt.title('LSTM-driven Trading Simulation VS Buy and Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(dash='dot'), name='Prophet Forecast'))
fig.add_trace(go.Scatter(x=lstm_dates, y=lstm_pred, mode='lines', line=dict(dash='dash'), name='LSTM Forecast'))
fig.add_trace(go.Scatter(x=arima_idx, y=arima_forecast, mode='lines', name='ARIMA Forecast'))
fig.add_trace(go.Scatter(x=sarima_idx, y=sarima_forecast, mode='lines', name='SARIMA Forecast'))
fig.add_trace(go.Scatter(x=df_anomalies.index, y=df_anomalies['Close'], mode='markers', name='Anomalies', marker=dict(color='red', size=8)))

fig.update_layout(
    title="Tesla Stock Price Forecast Comparison and Anomalies",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template='plotly_white'
)
fig.show()

