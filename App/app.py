from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from engine import (
    MODEL_FEATURES,
    RAW_REQUIRED_COLUMNS,
    build_analytics_tables,
    consumption_status,
    load_model,
    load_raw_data,
    prepare_analytics_data,
    prepare_model_data,
    prepare_uploaded_model_data,
    validate_raw_columns,
)


st.set_page_config(
    page_title="Smart Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "Dataset" / "household_power_consumption.txt")
MODEL_PATH = str(BASE_DIR / "Models" / "xgboost_energy_forecasting.pkl")

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fc; }
    .block-container { max-width: 1450px; padding-top: 1.6rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, .05);
    }
    h1, h2, h3 { color: #172033 !important; }
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #eef6ff 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 22px;
        box-shadow: 0 4px 18px rgba(37, 99, 235, .08);
    }
    .hero-title { color: #172033; font-size: 38px; font-weight: 800; }
    .hero-text { color: #526079; font-size: 16px; margin-top: 6px; }
    .section-title { color: #172033; font-size: 23px; font-weight: 750; margin: 14px 0 10px; }
    .small-note { color: #64748b; font-size: 13px; }
    .stButton > button { border-radius: 10px; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_startup():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    try:
        raw = load_raw_data(DATA_PATH)
        model_obj = load_model(MODEL_PATH)
        return raw, model_obj
    except Exception as exc:
        st.error(f"Project loading failed: {exc}")
        st.stop()


raw_df, model = safe_startup()

with st.sidebar:
    st.markdown("## ⚡ Energy AI")
    st.caption("Smart forecasting and monitoring")
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 Analytics",
            "🤖 Prediction",
            "⚡ Real-Time Simulation",
            "📁 Batch Prediction",
            "🧠 Model Insights",
            "📋 Data Overview",
        ],
    )
    st.divider()
    st.caption("Production model")
    st.write("XGBoost Regressor")
    st.caption("Target")
    st.write("Global_active_power")
    st.caption("Features")
    st.write(f"{len(MODEL_FEATURES)} engineered features")


if page == "🏠 Dashboard":
    analytics_df = prepare_analytics_data(DATA_PATH)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">⚡ Smart Energy Forecasting</div>
            <div class="hero-text">Household electricity analytics, prediction, monitoring, and historical real-time simulation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    avg_power = analytics_df["Global_active_power"].mean()
    max_power = analytics_df["Global_active_power"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(raw_df):,}")
    c2.metric("Average Power", f"{avg_power:.3f} kW")
    c3.metric("XGBoost R²", "97.21%")
    c4.metric("RMSE", "0.1499")

    st.markdown('<div class="section-title">📈 Consumption Trend</div>', unsafe_allow_html=True)
    trend_df = analytics_df.iloc[:5000]
    fig = px.line(
        trend_df,
        x="Datetime",
        y="Global_active_power",
        labels={"Global_active_power": "Active Power (kW)"},
        title="First 5,000 Valid Observations",
    )
    fig.update_layout(template="plotly_white", height=430, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    left, right = st.columns(2)
    with left:
        sample = analytics_df["Global_active_power"].head(12000)
        fig = px.histogram(sample, x="Global_active_power", nbins=55, title="Power Distribution")
        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        meter_data = pd.DataFrame(
            {
                "Meter": ["Sub Meter 1", "Sub Meter 2", "Sub Meter 3"],
                "Average": [
                    analytics_df["Sub_metering_1"].mean(),
                    analytics_df["Sub_metering_2"].mean(),
                    analytics_df["Sub_metering_3"].mean(),
                ],
            }
        )
        fig = px.bar(meter_data, x="Meter", y="Average", title="Average Sub-Metering")
        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.info(f"Maximum recorded active power in the cleaned analytics data: {max_power:.3f} kW")


elif page == "📊 Analytics":
    tables = build_analytics_tables(DATA_PATH)
    analytics_df = prepare_analytics_data(DATA_PATH)

    st.title("📊 Energy Analytics")
    st.caption("Cached aggregated tables keep page switching fast.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🕐 Hourly", "📅 Monthly", "📆 Weekday", "📉 Daily Trend", "📊 Distribution"]
    )

    with tab1:
        fig = px.line(
            tables["hourly"],
            x="Hour",
            y="Global_active_power",
            markers=True,
            title="Average Consumption by Hour",
        )
        fig.update_layout(template="plotly_white", height=490)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            tables["monthly"],
            x="Month",
            y="Global_active_power",
            title="Average Consumption by Month",
        )
        fig.update_layout(template="plotly_white", height=490)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.bar(
            tables["weekday"],
            x="Weekday",
            y="Global_active_power",
            title="Average Consumption by Weekday",
        )
        fig.update_layout(template="plotly_white", height=490)
        st.plotly_chart(fig, use_container_width=True)

        weekend_labels = tables["weekend"].copy()
        weekend_labels["Period"] = weekend_labels["Is_Weekend"].map({0: "Weekday", 1: "Weekend"})
        fig = px.bar(
            weekend_labels,
            x="Period",
            y="Global_active_power",
            title="Weekday vs Weekend",
        )
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        daily_long = tables["daily"].melt(
            id_vars="Datetime",
            value_vars=["Global_active_power", "Rolling_30_Day"],
            var_name="Series",
            value_name="Power",
        )
        fig = px.line(
            daily_long,
            x="Datetime",
            y="Power",
            color="Series",
            title="Daily Average and 30-Day Rolling Mean",
        )
        fig.update_layout(template="plotly_white", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        sample = analytics_df["Global_active_power"].head(20000)
        fig = px.histogram(
            sample,
            x="Global_active_power",
            nbins=60,
            marginal="box",
            title="Energy Consumption Distribution",
        )
        fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)


elif page == "🤖 Prediction":
    X_app, metadata = prepare_model_data(DATA_PATH)

    st.title("🤖 Energy Consumption Prediction")
    st.caption("Uses the exact 28-feature order from model training.")

    if X_app.empty:
        st.error("No model-ready observations were generated.")
        st.stop()

    mode = st.radio(
        "Prediction Mode",
        ["Latest Available Record", "Select Historical Record"],
        horizontal=True,
    )

    if mode == "Latest Available Record":
        selected_index = len(X_app) - 1
    else:
        selected_index = st.slider(
            "Historical observation index",
            min_value=0,
            max_value=len(X_app) - 1,
            value=len(X_app) - 1,
        )

    selected_features = X_app.iloc[[selected_index]]
    selected_info = metadata.iloc[selected_index]

    if st.button("⚡ Generate Prediction", type="primary", use_container_width=True):
        prediction = float(model.predict(selected_features)[0])
        actual = float(selected_info["Global_active_power"])
        error = abs(actual - prediction)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted", f"{prediction:.3f} kW")
        c2.metric("Actual", f"{actual:.3f} kW")
        c3.metric("Absolute Error", f"{error:.3f} kW")
        c4.metric("Status", consumption_status(prediction))
        st.caption(f"Timestamp: {selected_info['Datetime']}")

        with st.expander("View exact model inputs"):
            display_features = selected_features.T.reset_index()
            display_features.columns = ["Feature", "Value"]
            st.dataframe(display_features, use_container_width=True, hide_index=True)


elif page == "⚡ Real-Time Simulation":
    X_app, metadata = prepare_model_data(DATA_PATH)

    st.title("⚡ Historical Real-Time Simulation")
    st.warning("This replays historical records like a live stream; it is not a connected IoT smart meter.")

    if "simulation_index" not in st.session_state:
        st.session_state.simulation_index = max(0, len(X_app) - 500)

    start_index = max(0, len(X_app) - 500)
    max_index = len(X_app) - 1

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("⏮ Previous", use_container_width=True):
        st.session_state.simulation_index = max(start_index, st.session_state.simulation_index - 1)
    if b2.button("▶ Next", type="primary", use_container_width=True):
        st.session_state.simulation_index = min(max_index, st.session_state.simulation_index + 1)
    jump = b3.number_input("Jump", min_value=1, max_value=100, value=10)
    if b4.button("⏩ Jump", use_container_width=True):
        st.session_state.simulation_index = min(max_index, st.session_state.simulation_index + int(jump))
    if st.button("🔄 Reset"):
        st.session_state.simulation_index = start_index

    idx = st.session_state.simulation_index
    current_features = X_app.iloc[[idx]]
    current_info = metadata.iloc[idx]
    current_prediction = float(model.predict(current_features)[0])
    current_actual = float(current_info["Global_active_power"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual", f"{current_actual:.3f} kW")
    c2.metric("Predicted", f"{current_prediction:.3f} kW")
    c3.metric("Error", f"{abs(current_actual-current_prediction):.3f} kW")
    c4.metric("Status", consumption_status(current_prediction))
    st.caption(f"Simulation timestamp: {current_info['Datetime']}")

    window_start = max(start_index, idx - 99)
    window_features = X_app.iloc[window_start : idx + 1]
    window_info = metadata.iloc[window_start : idx + 1]
    window_predictions = model.predict(window_features)

    chart_df = pd.DataFrame(
        {
            "Datetime": window_info["Datetime"].values,
            "Actual": window_info["Global_active_power"].values,
            "Predicted": window_predictions,
        }
    )
    long_df = chart_df.melt(
        id_vars="Datetime",
        value_vars=["Actual", "Predicted"],
        var_name="Series",
        value_name="Power",
    )
    fig = px.line(long_df, x="Datetime", y="Power", color="Series", title="Live Actual vs Predicted")
    fig.update_layout(template="plotly_white", height=510, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.progress((idx - start_index) / max(1, max_index - start_index))


elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")
    st.caption("Upload raw CSV/TXT data with the original nine columns.")

    uploaded_file = st.file_uploader("Upload CSV or semicolon-separated TXT", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            separator = ";" if uploaded_file.name.lower().endswith(".txt") else ","
            uploaded_df = pd.read_csv(uploaded_file, sep=separator, low_memory=False)
            missing = validate_raw_columns(uploaded_df)

            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            elif len(uploaded_df) < 1441:
                st.error("At least 1,441 chronological rows are required for Lag_1440.")
            else:
                st.success(f"Loaded {len(uploaded_df):,} rows")
                st.dataframe(uploaded_df.head(20), use_container_width=True)

                if st.button("Generate Batch Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        uploaded_X, uploaded_metadata = prepare_uploaded_model_data(uploaded_df)
                        predictions = model.predict(uploaded_X)
                        result = uploaded_metadata.copy()
                        result["Predicted_Global_active_power"] = predictions
                        result["Absolute_Error"] = np.abs(
                            result["Global_active_power"] - result["Predicted_Global_active_power"]
                        )

                    st.success(f"Generated {len(result):,} predictions")
                    st.dataframe(result.head(100), use_container_width=True)
                    st.download_button(
                        "⬇ Download Predictions CSV",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="energy_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"File processing failed: {exc}")


elif page == "🧠 Model Insights":
    st.title("🧠 Model Performance & Insights")

    performance = pd.DataFrame(
        {
            "Model": ["Random Forest", "XGBoost"],
            "MAE": [0.0589, 0.0620],
            "RMSE": [0.1596, 0.1499],
            "R2": [0.9683, 0.9721],
            "MAPE (%)": [6.41, 6.72],
            "Training Time (sec)": [736.30, 79.72],
        }
    )
    st.dataframe(performance, use_container_width=True, hide_index=True)

    tab1, tab2, tab3 = st.tabs(["R²", "RMSE", "Training Time"])
    with tab1:
        fig = px.bar(performance, x="Model", y="R2", text_auto=".4f", title="R² Comparison")
        fig.update_layout(template="plotly_white", yaxis_range=[0.90, 1.00])
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.bar(performance, x="Model", y="RMSE", text_auto=".4f", title="RMSE Comparison")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = px.bar(
            performance,
            x="Model",
            y="Training Time (sec)",
            text_auto=".2f",
            title="Training Time Comparison",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("XGBoost Feature Importance")
    feature_importance = pd.DataFrame(
        {"Feature": MODEL_FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    top = feature_importance.head(15).sort_values("Importance")
    fig = px.bar(top, x="Importance", y="Feature", orientation="h", title="Top 15 Features")
    fig.update_layout(template="plotly_white", height=610)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Lag_1 dominates because household power consumption is strongly autocorrelated minute-to-minute.")


elif page == "📋 Data Overview":
    st.title("📋 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Duplicate Rows", f"{raw_df.duplicated().sum():,}")

    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(100), use_container_width=True, height=390)

    left, right = st.columns(2)
    with left:
        st.subheader("Data Types")
        dtype_df = pd.DataFrame(
            {"Column": raw_df.columns, "Data Type": raw_df.dtypes.astype(str).values}
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Missing Values")
        missing = raw_df.isnull().sum()
        missing_df = pd.DataFrame({"Column": missing.index, "Missing Values": missing.values})
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.caption("Raw question-mark placeholders may not appear as NaN until numeric conversion is performed.")
