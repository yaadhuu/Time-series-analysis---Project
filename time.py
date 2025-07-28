import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


st.set_page_config(page_title="General Time Series Analysis", layout='wide', initial_sidebar_state='expanded')
st.markdown("""
<style>
    body, .stApp {background-color:#2D323E;}
    .stApp {color: #E0E0E0;}
    .css-1n76uvr {background: #23272f;}
    .st-dj {color: #E0E0E0;}
    .st-ef {background: #2D323E;}
    .st-cb, .st-at {background: #23272f; color:#E0E0E0;}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Dataset Identifier (e.g., Stock Ticker: AAPL, BTC-USD)", 'AAPL').upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today').date())
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 120, 60)

st.sidebar.subheader("Models to Include")
run_prophet = st.sidebar.checkbox("Prophet Model", True)
run_lstm = st.sidebar.checkbox("LSTM Model", True)
run_arima = st.sidebar.checkbox("ARIMA Model", True)
run_sarima = st.sidebar.checkbox("SARIMA Model", True)
run_garch = st.sidebar.checkbox("GARCH Volatility Model", True)
run_anomaly = st.sidebar.checkbox("Show Anomalies", True)


app_title_placeholder = st.empty()
app_title_placeholder.title("Time Series Analysis")



@st.cache_data(show_spinner="Fetching and preparing historical data...")
def get_data(ticker_symbol, start_dt, end_dt):
    if not ticker_symbol or ticker_symbol.strip() == "":
        return pd.DataFrame(), False, False, "Please enter a dataset identifier (e.g., Stock Ticker)."
    try:
        df_raw = yf.download(ticker_symbol, start=start_dt, end=end_dt, auto_adjust=False)
        if df_raw.empty:
            return pd.DataFrame(), False, False, f"No data downloaded for '{ticker_symbol}'."
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = ['_'.join([str(i) for i in col if i]).strip().replace(' ', '') for col in
                              df_raw.columns.values]
        df_raw.reset_index(inplace=True)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw.set_index('Date', inplace=True)
        close_candidates = [f'AdjClose_{ticker_symbol}', 'AdjClose', f'Close_{ticker_symbol}', 'Close']
        close_col = next((col for col in close_candidates if col in df_raw.columns), None)
        if close_col is None:
            return pd.DataFrame(), False, False, f"Could not find a suitable 'Close' or 'Adj Close' column for '{ticker_symbol}'."
        df_raw.rename(columns={close_col: 'Close'}, inplace=True)
        ohlcv_cols = ['Open', 'High', 'Low', 'Volume']
        has_ohlcv = all(col in df_raw.columns for col in ohlcv_cols)
        has_volume = 'Volume' in df_raw.columns and df_raw['Volume'].sum() > 0
        for col in ohlcv_cols:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        subset = ['Close'] + [c for c in ohlcv_cols if c in df_raw.columns]
        df_raw.dropna(subset=subset, inplace=True)
        if df_raw.empty:
            return pd.DataFrame(), False, False, f"No valid time series data for '{ticker_symbol}' after cleaning."
        return df_raw, has_ohlcv, has_volume, None
    except Exception as e:
        return pd.DataFrame(), False, False, f"Error fetching/processing data for '{ticker_symbol}': {e}"



df, has_ohlcv, has_volume, error_message = get_data(ticker, start_date, end_date)


if error_message:
    st.error(error_message)
    app_title_placeholder.title("Time Series Analysis - Error")
    st.stop()
elif df.empty or len(df) < 120:
    st.warning(f"Insufficient data for '{ticker}'. At least 120 days of historical data required.")
    app_title_placeholder.title("Time Series Analysis - Insufficient Data")
    st.stop()
else:
    app_title_placeholder.title(f" Time Series Analysis for {ticker}")


    df['Naive'] = df['Close'].shift(1)
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Return'].rolling(window=20).std()
    if has_ohlcv and 'MA_20' in df.columns:
        df['BB_Upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
    df.dropna(inplace=True)
    if df.empty or len(df) < 120:
        st.warning("Not enough data after preprocessing.")
        st.stop()


    anoms = []
    if run_anomaly and len(df) >= 30:
        rolling_mean = df['Close'].rolling(30).mean()
        rolling_std = df['Close'].rolling(30).std().replace(0, np.nan)
        z_score = (df['Close'] - rolling_mean) / rolling_std
        df['Anomaly'] = ((z_score.abs() > 2.5) & ~z_score.isna()).astype(int)
        anoms = df[df['Anomaly'] == 1].index


    prophet_dates, prophet_pred, forecast_prophet_df = [], [], None
    if run_prophet:
        df_prophet = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet.dropna(subset=['y'], inplace=True)
        if not df_prophet.empty:
            try:
                prophet = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
                prophet.fit(df_prophet)
                future = prophet.make_future_dataframe(periods=forecast_days, include_history=False)
                forecast_prophet_df = prophet.predict(future)
                prophet_dates = forecast_prophet_df['ds']
                prophet_pred = forecast_prophet_df['yhat']
            except Exception as e:
                st.warning(f"Prophet failed: {e}")


    lstm_dates, lstm_pred, y_true_lstm_eval = [], [], []
    if run_lstm and len(df) >= 120:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close']])
        train_sz = int(len(scaled) * 0.8)
        if train_sz >= 60 and (len(scaled) - train_sz) >= 60:
            train, test = scaled[:train_sz], scaled[train_sz - 60:]


            def create_sequences(data, step=60):
                X, y = [], []
                for i in range(step, len(data)):
                    X.append(data[i - step:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)


            Xtr, ytr = create_sequences(train, 60)
            Xte, yte = create_sequences(test, 60)
            Xtr = Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1))
            Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
            try:
                lstm = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(Xtr.shape[1], 1)),
                    LSTM(50),
                    Dense(1)
                ])
                lstm.compile(loss='mean_squared_error', optimizer='adam')
                lstm.fit(Xtr, ytr, epochs=8, batch_size=32, verbose=0)
                pred_scaled = lstm.predict(Xte)
                lstm_pred = scaler.inverse_transform(pred_scaled).flatten()
                y_true_lstm_eval = scaler.inverse_transform(yte.reshape(-1, 1)).flatten()
                lstm_dates = df.index[train_sz:][:len(lstm_pred)]
            except Exception as e:
                st.warning(f"LSTM failure: {e}")


    arima_idx, arima_pred = [], []
    if run_arima and len(df) >= 50:
        try:
            arima_mod = ARIMA(df['Close'], order=(5, 1, 0)).fit()
            arima_pred = arima_mod.forecast(steps=forecast_days)
            arima_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        except Exception as e:
            st.warning(f"ARIMA failure: {e}")


    sarima_idx, sarima_pred = [], []
    if run_sarima and len(df) >= 60:
        try:
            sarima_mod = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            sarima_pred = sarima_mod.forecast(steps=forecast_days)
            sarima_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        except Exception as e:
            st.warning(f"SARIMA failure: {e}")


    vol_pred = []
    if run_garch:
        rets = 100 * df['Close'].pct_change().dropna()
        if len(rets) >= 60:
            try:
                garch_mod = arch_model(rets, vol='Garch', p=1, q=1)
                garch_fit = garch_mod.fit(disp='off')
                vforecast = garch_fit.forecast(horizon=forecast_days)
                if vforecast.variance is not None and not vforecast.variance.empty:
                    vol_pred = np.sqrt(vforecast.variance.iloc[-1, :])
            except Exception as e:
                st.warning(f"GARCH failure: {e}")


    st.subheader(f"{ticker} Key Metrics")
    current_price = df['Close'].iloc[-1]
    previous_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
    daily_change = current_price - previous_close
    daily_pct_change = (daily_change / previous_close) * 100 if previous_close != 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Value", f"${current_price:,.2f}")
    c2.metric("Daily Change", f"${daily_change:,.2f}", delta=f"{daily_pct_change:,.2f}%")
    c3.metric("Latest Volume", f"{df['Volume'].iloc[-1]:,.0f}" if has_volume else "N/A")
    c4.metric("Avg Daily Volatility (20D)", f"{df['Volatility_20'].iloc[-1] * 100:.2f}%")

    st.markdown("---")


    st.header(f"{ticker} Value & Forecasts")

    if has_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.08)


    if has_ohlcv:
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                     name='Actual Value', increasing_line_color='#4CAF50',
                                     decreasing_line_color='#FF5555'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Value', mode='lines',
                                 line=dict(color='#1E90FF', width=2)), row=1, col=1)


    if has_ohlcv and all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'MA_20']):
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name='MA (20)', line=dict(color='#CCCC00', width=1)), row=1,
                      col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='#800080', width=1, dash='dot')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='#800080', width=1, dash='dot'),
                       fill='tonexty', fillcolor='rgba(128,0,128,0.05)'), row=1, col=1)


    if run_prophet and len(prophet_dates) > 0:
        fig.add_trace(go.Scatter(x=prophet_dates, y=prophet_pred, name='Prophet Forecast',
                                 mode='lines', line=dict(dash='dash', color='#FFD700', width=2)), row=1, col=1)
    if run_lstm and len(lstm_dates) > 0:
        fig.add_trace(go.Scatter(x=lstm_dates, y=lstm_pred, name='LSTM Forecast',
                                 mode='lines', line=dict(dash='dot', color='#00FFFF', width=2)), row=1, col=1)
    if run_arima and len(arima_idx) > 0:
        fig.add_trace(go.Scatter(x=arima_idx, y=arima_pred, name='ARIMA Forecast',
                                 mode='lines', line=dict(color='#A020F0', width=2)), row=1, col=1)
    if run_sarima and len(sarima_idx) > 0:
        fig.add_trace(go.Scatter(x=sarima_idx, y=sarima_pred, name='SARIMA Forecast',
                                 mode='lines', line=dict(color='#FFA500', width=2)), row=1, col=1)


    if has_ohlcv and len(df) > 180:
        recent_low = df['Low'].iloc[-180:].min()
        recent_high = df['High'].iloc[-180:].max()
        fig.add_shape(type="line", x0=df.index[-180], y0=recent_low, x1=df.index[-1], y1=recent_low,
                      line=dict(color="#FF4500", width=1, dash="dashdot"))
        fig.add_annotation(x=df.index[-100], y=recent_low + (df['Close'].max() - df['Close'].min()) * 0.01,
                           text="Support", showarrow=False, font=dict(color="#FF4500", size=10))
        fig.add_shape(type="line", x0=df.index[-180], y0=recent_high, x1=df.index[-1], y1=recent_high,
                      line=dict(color="#1E90FF", width=1, dash="dashdot"))
        fig.add_annotation(x=df.index[-100], y=recent_high - (df['Close'].max() - df['Close'].min()) * 0.01,
                           text="Resistance", showarrow=False, font=dict(color="#1E90FF", size=10))


    if run_anomaly and len(anoms) > 0:
        for anomaly_date in anoms:
            fig.add_annotation(
                x=anomaly_date,
                y=df.loc[anomaly_date, 'Close'],
                xref="x",
                yref="y",
                text="Anomaly",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#FF5555",
                ax=0,
                ay=-40,
                font=dict(size=12, color="#FF5555"),
                bgcolor="rgba(255,85,85,0.1)",
                bordercolor="#FF5555",
                borderwidth=1,
                borderpad=4
            )


    if has_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='#606060'), row=2, col=1)


    if has_volume:
        fig.update_layout(
            title=f'Historical Value, Volume & Forecast for {ticker}',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#2D323E',
            plot_bgcolor='#2D323E',
            font=dict(color='#E0E0E0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=80, b=20),
            height=700,
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.05, bordercolor='#4F566A', bgcolor='#3C4251'),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ],
                    font=dict(color='#E0E0E0'), bgcolor='#3C4251', activecolor='#1E90FF', bordercolor='#4F566A'
                ),
            ),
            xaxis2=dict(rangeslider=dict(visible=False))
        )
        fig.update_yaxes(title_text='Value (USD)', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
    else:
        fig.update_layout(
            title=f'Historical Value & Forecast for {ticker}',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#2D323E',
            plot_bgcolor='#2D323E',
            font=dict(color='#E0E0E0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.05, bordercolor='#4F566A', bgcolor='#3C4251'),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ],
                    font=dict(color='#E0E0E0'), bgcolor='#3C4251', activecolor='#1E90FF', bordercolor='#4F566A'
                ),
            )
        )
        fig.update_yaxes(title_text='Value (USD)', row=1, col=1)
        fig.update_xaxes(title_text='Date', row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


    st.subheader(f" {ticker} Forecasting Outlook")
    last_hist_date = df.index[-1]
    actual_future_dates = pd.date_range(last_hist_date + pd.Timedelta(days=1), periods=forecast_days)


    def calculate_metrics(model_name, preds, pred_dates):
        common_dates = actual_future_dates.intersection(pred_dates)
        actuals = df['Close'].reindex(common_dates).dropna()
        if not actuals.empty and len(preds) == len(pred_dates):
            preds_series = pd.Series(preds, index=pred_dates).reindex(actuals.index).dropna()
            if len(actuals) == len(preds_series):
                rmse = np.sqrt(mean_squared_error(actuals, preds_series))
                mae = mean_absolute_error(actuals, preds_series)
                return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'Predictions': preds_series}
        return None


    models_metrics = []
    if run_prophet and forecast_prophet_df is not None and not forecast_prophet_df.empty:
        future_pred = forecast_prophet_df[forecast_prophet_df['ds'] > last_hist_date].copy()
        metric_row = calculate_metrics('Prophet', list(future_pred['yhat']), future_pred['ds'])
        if metric_row: models_metrics.append(metric_row)

    if run_lstm and len(lstm_pred) > 0 and len(y_true_lstm_eval) > 0:
        rmse = np.sqrt(mean_squared_error(y_true_lstm_eval, lstm_pred))
        mae = mean_absolute_error(y_true_lstm_eval, lstm_pred)
        lstm_series = pd.Series(lstm_pred, index=lstm_dates)
        models_metrics.append({'Model': 'LSTM', 'RMSE': rmse, 'MAE': mae, 'Predictions': lstm_series})

    if run_arima and len(arima_pred) > 0 and len(arima_idx) > 0:
        metric_row = calculate_metrics('ARIMA', arima_pred, arima_idx)
        if metric_row: models_metrics.append(metric_row)

    if run_sarima and len(sarima_pred) > 0 and len(sarima_idx) > 0:
        metric_row = calculate_metrics('SARIMA', sarima_pred, sarima_idx)
        if metric_row: models_metrics.append(metric_row)

    best_model = None
    if models_metrics:
        df_metrics = pd.DataFrame(models_metrics)
        df_metrics = df_metrics.dropna(subset=['RMSE'])
        if not df_metrics.empty:
            best_idx = df_metrics['RMSE'].idxmin()
            best_model_row = df_metrics.loc[best_idx]
            best_model = best_model_row['Model']
            st.write(
                f"The **{best_model}** model appears to be most accurate (RMSE: **{best_model_row['RMSE']:.4f}**).")
            if not best_model_row['Predictions'].empty:
                st.subheader(f"Next {min(5, len(best_model_row['Predictions']))} Days Forecast ({best_model})")
                display = best_model_row['Predictions'].head(5).reset_index()
                display.columns = ['Date', 'Predicted Value']
                display['Date'] = display['Date'].dt.strftime('%Y-%m-%d')
                display['Predicted Value'] = display['Predicted Value'].apply(lambda x: f'${x:,.2f}')
                st.table(display)

                fig_out = go.Figure()
                hist = df['Close'].tail(30)
                forecast_data = best_model_row['Predictions'].head(forecast_days)
                fig_out.add_trace(go.Scatter(x=hist.index, y=hist, mode='lines', name='Recent Actuals',
                                             line=dict(color='#88CCEE', width=2)))
                fig_out.add_trace(
                    go.Scatter(x=forecast_data.index, y=forecast_data, mode='lines', name=f'{best_model} Forecast',
                               line=dict(color='#FFD700', width=2, dash='dash')))
                fig_out.update_layout(title=f'Short-term Outlook for {ticker}', xaxis_title='Date', yaxis_title='Value',
                                      hovermode='x unified', template='plotly_dark', paper_bgcolor='#2D323E',
                                      plot_bgcolor='#2D323E', font=dict(color='#E0E0E0'), legend=dict(orientation="h"),
                                      margin=dict(l=20, r=20, t=80, b=20), height=350)
                st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.info("No model evaluation metrics available.")
    else:
        st.info("No models with forecast metrics for evaluation.")

    st.markdown("---")
    st.subheader("All Model Performance Metrics")

    if models_metrics:
        metrics_table = [{'Model': m['Model'], 'RMSE': f"{m['RMSE']:.4f}", 'MAE': f"{m['MAE']:.4f}"} for m in
                         models_metrics]
        st.dataframe(pd.DataFrame(metrics_table).set_index('Model'), use_container_width=True)
    else:
        st.info("No models able to generate evaluation metrics.")

    st.markdown("""
    ---
    <p style='text-align: center; color: #888888; font-size: 0.9em;'>
    Developed by Yadhu Krishna
    </p>
    """, unsafe_allow_html=True)
