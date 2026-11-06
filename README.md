# ⚡ Smart Energy Consumption Forecasting & Monitoring System

An end-to-end **Machine Learning-based Energy Consumption Forecasting System** developed using Python, Random Forest, XGBoost, and Streamlit.

The project analyzes more than **2 million minute-level household electricity consumption records**, performs time-series feature engineering, compares multiple regression models, and provides an interactive web application for energy analytics, prediction, batch processing, and historical-data-based real-time simulation.

---

## 📌 Project Overview

Accurate electricity consumption forecasting can support energy planning, smart-grid management, and efficient use of electrical resources.

This project develops a machine learning pipeline for predicting household **Global Active Power** using historical electricity consumption data.

The complete workflow includes:

- Data preprocessing and validation
- Exploratory Data Analysis (EDA)
- Time-series feature engineering
- Lag and rolling-window features
- Random Forest and XGBoost regression
- Model evaluation and comparison
- Feature importance analysis
- Model serialization
- Interactive Streamlit dashboard
- Historical-data-based real-time simulation
- Batch prediction

> **Note:** The current implementation is a prototype based on historical household electricity data. It is not connected to a live smart meter or utility grid.

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Analyze household electricity consumption patterns
- Prepare a large time-series dataset for machine learning
- Engineer meaningful temporal, lag, and rolling features
- Predict household active power consumption
- Compare Random Forest and XGBoost regression models
- Evaluate models using multiple regression metrics
- Identify important predictors of electricity consumption
- Develop an interactive dashboard for prediction and monitoring
- Demonstrate how the approach could be extended to smart-meter and energy-management applications

---

## 📊 Dataset

The project uses the **Individual Household Electric Power Consumption Dataset**.

The dataset contains more than **2 million observations** recorded at approximately one-minute intervals.

### Original Dataset Features

| Feature | Description |
|---|---|
| `Date` | Date of measurement |
| `Time` | Time of measurement |
| `Global_active_power` | Household global active power in kilowatts |
| `Global_reactive_power` | Household global reactive power |
| `Voltage` | Average voltage |
| `Global_intensity` | Average electrical current intensity |
| `Sub_metering_1` | Energy sub-metering measurement 1 |
| `Sub_metering_2` | Energy sub-metering measurement 2 |
| `Sub_metering_3` | Energy sub-metering measurement 3 |

### Prediction Target

```text
Global_active_power
```

The objective of the ML models is to estimate household active power consumption.

> The complete dataset is not included in this repository because of its large size.

---

## 🔄 Project Workflow

```text
Historical Electricity Dataset
            │
            ▼
     Data Understanding
            │
            ▼
      Data Cleaning
            │
            ▼
Exploratory Data Analysis
            │
            ▼
   Feature Engineering
            │
            ▼
Chronological Train/Test Split
            │
            ▼
 ┌───────────────────────┐
 │     Random Forest     │
 │       XGBoost         │
 └───────────────────────┘
            │
            ▼
   Model Evaluation
            │
            ▼
  Model Comparison
            │
            ▼
 Feature Importance
            │
            ▼
     Model Saving
            │
            ▼
 Streamlit Application
            │
            ▼
Analytics + Prediction + Simulation
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Dataset structure inspection
- Missing-value analysis
- Conversion of numerical columns to appropriate data types
- Invalid-value handling
- Duplicate-record checking
- Combination of `Date` and `Time` into a single datetime representation
- Chronological sorting
- Outlier analysis
- Preparation of data for time-dependent feature engineering

Valid high-consumption observations were preserved rather than automatically removing all statistical outliers.

---

## 📈 Exploratory Data Analysis

Several analyses were performed to understand electricity consumption behavior, including:

- Electricity consumption distribution
- Time-series consumption trends
- Hourly consumption patterns
- Monthly consumption patterns
- Weekday analysis
- Weekday vs weekend comparison
- Correlation analysis
- Outlier analysis
- Sub-metering analysis
- Rolling consumption trends

---

## ⚙️ Feature Engineering

Time-series and calendar-based features were generated to help the models learn temporal consumption patterns.

### 🕐 Calendar Features

```text
Year
Month
Day
Hour
Minute
Weekday
Is_Weekend
Peak_Hour
Time_Period
```

The day was also divided into periods such as:

- Morning
- Afternoon
- Evening
- Night

Categorical variables were transformed using one-hot encoding.

---

### ⏳ Lag Features

Historical consumption was incorporated through lag features:

| Feature | Meaning |
|---|---|
| `Lag_1` | Previous minute's active power |
| `Lag_60` | Active power approximately one hour earlier |
| `Lag_1440` | Active power approximately one day earlier |

These features allow the model to learn short-term and daily dependencies in electricity consumption.

---

### 📉 Rolling Features

Rolling-window statistics were generated over recent observations:

```text
Rolling_Mean_60
Rolling_STD_60
Rolling_Max_60
Rolling_Min_60
```

The rolling calculations were based on shifted historical consumption to avoid directly using the target observation in its own rolling statistics.

---

## 🔐 Leakage Consideration

During model development, `Global_intensity` showed extremely dominant feature importance because of its very strong physical relationship with active power.

To create a more meaningful forecasting setup and reduce target-proxy behavior, `Global_intensity` was excluded from the final model feature set.

---

## 🤖 Machine Learning Models

Two regression algorithms were evaluated.

### 1. Random Forest Regressor

Random Forest provides an ensemble of decision trees and was used as a strong machine-learning baseline.

### 2. XGBoost Regressor

XGBoost uses gradient-boosted decision trees and was evaluated for improved predictive performance and computational efficiency.

---

## 📊 Model Performance

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Random Forest | **0.0589** | 0.1596 | 0.9683 | **6.41%** |
| XGBoost | 0.0620 | **0.1499** | **0.9721** | 6.72% |

### Training Performance

| Model | Training Time | Prediction Time |
|---|---:|---:|
| Random Forest | 736.30 sec | 1.97 sec |
| XGBoost | **79.72 sec** | 2.07 sec |

---

## 🏆 Final Model — XGBoost

**XGBoost** was selected as the final deployment model.

It achieved:

- **R² = 0.9721**
- **RMSE = 0.1499**
- MAE = 0.0620
- MAPE = 6.72%
- Significantly faster training than Random Forest

Although Random Forest achieved slightly better MAE and MAPE, XGBoost provided the stronger overall balance of **R², RMSE, and training efficiency**.

The trained model is serialized as:

```text
xgboost_energy_forecasting.pkl
```

---

## 🧠 Feature Importance

The most influential XGBoost features included:

1. `Lag_1`
2. `Rolling_Mean_60`
3. `Sub_metering_1`
4. `Sub_metering_3`
5. `Sub_metering_2`
6. `Rolling_Max_60`
7. `Rolling_Min_60`
8. `Time_Period_Evening`
9. `Voltage`
10. `Peak_Hour`

`Lag_1` was the strongest feature, demonstrating the strong short-term dependency present in minute-level household electricity consumption.

---

## ⏱️ Time-Series Forecasting Approach

This project uses a **machine-learning-based time-series forecasting approach**.

Instead of using classical forecasting algorithms such as ARIMA or SARIMA, historical temporal information is transformed into supervised ML features through:

- Lag variables
- Rolling-window statistics
- Calendar variables
- Time-of-day information
- Chronological model evaluation

The system is primarily designed for **short-horizon, minute-level energy consumption prediction**.

---

# 🌐 Streamlit Web Application

The trained XGBoost model is integrated into an interactive **Streamlit application**.

The application provides seven major sections.

---

## 🏠 Dashboard

Provides a high-level overview of:

- Dataset size
- Average electricity consumption
- XGBoost performance
- Consumption trends
- Consumption distribution
- Sub-metering behavior

---

## 📊 Analytics

Interactive visualizations include:

- Hourly average consumption
- Monthly average consumption
- Weekday consumption
- Weekday vs weekend comparison
- Daily trends
- Rolling trends
- Consumption distributions

---

## 🤖 Prediction

The prediction interface allows users to evaluate historical observations using the trained XGBoost model.

It displays:

- Predicted active power
- Actual active power
- Absolute prediction error
- Consumption status
- Observation timestamp
- Model input features

---

## ⚡ Historical Real-Time Simulation

The application includes a real-time-style simulation that sequentially replays historical electricity records.

It displays:

- Actual consumption
- Predicted consumption
- Prediction error
- Consumption status
- Recent Actual vs Predicted trends

> This feature is a **historical-data-based real-time simulation** and does not represent a live IoT or smart-meter connection.

---

## 📁 Batch Prediction

Users can upload compatible electricity data in CSV/TXT format.

The system:

1. Validates the input data
2. Performs feature engineering
3. Generates XGBoost predictions
4. Calculates prediction errors when actual values are available
5. Allows results to be downloaded

---

## 🧠 Model Insights

The application provides:

- Random Forest vs XGBoost comparison
- R² comparison
- RMSE comparison
- Training-time comparison
- XGBoost feature-importance visualization

---

## 📋 Data Overview

Provides information about:

- Dataset dimensions
- Column information
- Data types
- Missing values
- Duplicate records
- Raw data preview

---

# 📁 Project Structure

```text
Energy Consumption Forecasting/
│
├── App/
│   ├── app.py
│   ├── engine.py
│   └── requirements.txt
│
├── Notebook/
│   └── Energy_Forecasting_ML.ipynb
│
├── Models/
│   └── xgboost_energy_forecasting.pkl
│
├── Images/
│   ├── dashboard.png
│   ├── analytics.png
│   ├── prediction.png
│   ├── simulation.png
│   └── feature_importance.png
│
├── Dataset/
│   └── household_power_consumption.txt
│
├── README.md
└── .gitignore
```

> The dataset may be excluded from GitHub due to its large file size.

---

# 💻 Technology Stack

### Programming & Data Processing

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Random Forest
- XGBoost

### Data Visualization

- Matplotlib
- Seaborn
- Plotly

### Development

- Jupyter Notebook
- Anaconda

### Web Application

- Streamlit

### Model Persistence

- Joblib

---


# 💡 Potential Real-World Applications

The techniques demonstrated in this project can be extended to:

- ⚡ Electricity demand forecasting
- 🏠 Smart-home energy management
- 🏢 Smart-building monitoring
- 🏭 Industrial energy monitoring
- 🔌 Smart-grid applications
- ☀️ Renewable-energy planning
- 📈 Peak-demand analysis
- 🚨 Energy anomaly detection

For example, a utility provider could adapt this approach using its own smart-meter and grid data to support electricity-demand planning.

---

# ⚠️ Current Limitations

The current implementation has several limitations:

- It uses historical data rather than a live smart-meter feed
- The dataset primarily represents a household rather than an entire electricity grid
- Weather information is not included
- Electricity pricing is not included
- Occupancy information is not available
- Appliance-state information is limited
- The system focuses primarily on short-horizon prediction
- Model performance on other households or geographic regions is not guaranteed

Therefore, this project should be considered an **ML forecasting and monitoring prototype**, rather than a production utility-grid system.

---

# 🔮 Future Improvements

Future development could include:

- Live smart-meter / IoT integration
- Weather and temperature features
- Multiple-household forecasting
- Geographic and regional demand forecasting
- Electricity-price integration
- Anomaly detection
- Automated model retraining
- Model-drift monitoring
- REST API development
- Cloud deployment
- Docker containerization
- CI/CD pipeline
- Advanced forecasting models such as LSTM, GRU, or Transformers

---

# 🎓 Key Learning Outcomes

This project demonstrates practical experience with:

- Large-scale dataset preprocessing
- Exploratory Data Analysis
- Time-series data handling
- Feature engineering
- Lag features
- Rolling-window statistics
- Target leakage analysis
- Machine-learning regression
- Random Forest
- XGBoost
- Regression evaluation metrics
- Feature importance
- Model serialization
- Streamlit application development
- Interactive data visualization
- Batch prediction
- Historical real-time simulation

---

# 👨‍💻 Author

**Muhammad Usman**

BS Computer Science  
Institute of Space Technology (IST), Islamabad

---

## ⭐ Repository Topics

`machine-learning` `data-science` `python` `time-series` `energy-forecasting` `xgboost` `random-forest` `streamlit` `plotly` `feature-engineering` `smart-grid` `jupyter-notebook`

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐.
