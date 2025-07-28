# Time Series Stock Market Forecasting Dashboard

## Internship Project — Zidio Development (Data Science & Analyst Intern)

---

## 📊 Project Overview

This project is a comprehensive **Time Series Forecasting Dashboard** developed as part of my **Data Science and Analyst Internship at Zidio Development**. The dashboard provides an interactive visualization and forecasting environment for financial time series data, enabling users to analyze historical patterns and predict future trends using multiple advanced models.

The project integrates multiple time series models — **Prophet, LSTM (Deep Learning), ARIMA, SARIMA, GARCH**, and **Anomaly Detection** — into a unified **Streamlit-based web application** with real-time visualization using Plotly.

---

## 🎯 Key Features

* 📈 **Dynamic Data Fetching**: Real-time data retrieval using Yahoo Finance (yfinance).
* 🧠 **Model Implementations**:

  * **Prophet (Facebook)** — Trend & seasonality detection with confidence intervals.
  * **LSTM (Bidirectional with EarlyStopping)** — Deep learning-based forecasting.
  * **ARIMA & SARIMA** — Classical statistical time series models.
  * **GARCH** — Volatility clustering for risk analysis.
  * **Anomaly Detection** — Z-score based outlier detection on price movements.
* 📊 **Visual Analytics**:

  * Candlestick OHLC charts with Bollinger Bands.
  * Overlay of model forecasts & confidence bands.
  * Anomaly markings on critical price deviations.
  * Support/Resistance level visualization.
* 🗃️ **Data Export**: Downloadable CSVs for model forecasted data.
* 🖥️ **Interactive Dashboard**: Developed using **Streamlit** with an intuitive sidebar control panel.

---

## 🛠️ Tech Stack

* **Python 3.10**
* **Streamlit** (Web App UI)
* **yfinance** (Data Source)
* **Prophet (Facebook Prophet)**
* **TensorFlow / Keras (LSTM)**
* **statsmodels (ARIMA/SARIMA)**
* **arch (GARCH Model)**
* **Plotly (Visualization)**
* **scikit-learn (Preprocessing & Metrics)**

---

## 🚀 Project Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yaadhuu/Time-series-analysis---Project
   cd zidio-time-series-dashboard
   ```

2. **Create Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

5. **Access the Dashboard**

   * Open browser and go to: `http://localhost:8501/`

---

## 📂 Repository Structure

```
├── app.py                # Main Streamlit App File
├── README.md              # Project Documentation
├── requirements.txt       # All Python Dependencies
└── assets/                # (Optional) Images, Logos, Supporting Files
```

---

## 📝 Learning & Contributions

During my internship at **Zidio Development**, I designed and built this project from scratch, gaining hands-on experience in:

* **Advanced Time Series Forecasting** techniques blending statistical and AI/ML models.
* Optimizing **LSTM models** with Bidirectional layers and EarlyStopping to prevent overfitting.
* Leveraging **Prophet's** seasonality and changepoint capabilities for trend forecasting.
* Applying **GARCH models** for volatility analysis and understanding financial risk modeling.
* Building end-to-end **interactive dashboards** for real-time data analysis and business intelligence.
* Performing **Model Performance Evaluation (RMSE, MAE)** to select the best-fit forecasting model.
* Enhancing the user experience with **professional UI/UX designs** using Streamlit & Plotly.


## 🤝 Acknowledgements

* **Zidio Development Team** for providing mentorship and an opportunity to explore real-world data analytics projects.
* Special thanks to my Internship Supervisor for guiding me through advanced forecasting models and business-driven analysis.

---

## 📬 Contact

**Yadhu Krishna P**
Email: [yeadhukrishna.p@gmail.com](mailto:yeadhukrishna.p@gmail.com)
LinkedIn: [linkedin.com/in/yadhu-krishna-p-6424972bb](https://linkedin.com/in/yadhu-krishna-p-6424972bb)
GitHub: [github.com/yaadhuu](https://github.com/yaadhuu)

---

## ⭐ Final Note

> This project serves as a foundation for building enterprise-level Financial Forecasting Dashboards for stock markets, commodities, and cryptocurrencies. Contributions and suggestions are welcome!

---

## License

This project is licensed under the MIT License.

---

### Developed under Zidio Development Internship | 2025
