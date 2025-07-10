import streamlit as st
import pandas as pd
import numpy as np
import os
from analytics_model import DataAnalyticsModel
from data_cleaner import DataCleaner
from data_testing_utils import DataTestingUtils
from advanced_analytics import AdvancedAnalytics
from dashboard_generator import DashboardGenerator

st.set_page_config(page_title="Super Data Analytics App", layout="wide")
st.title("📊 Super Data Analytics App")
st.markdown("""
A powerful, interactive analytics system for any kind of data. Upload your file, analyze, clean, and visualize your data quality instantly!
""")

# --- Sidebar ---
st.sidebar.header("Upload & Options")

uploaded_file = st.sidebar.file_uploader("Upload your data file (CSV, Excel, JSON, Parquet)", type=["csv", "xlsx", "xls", "json", "parquet"])

run_cleaning = st.sidebar.checkbox("Run Data Cleaning", value=True)
run_testing = st.sidebar.checkbox("Run Data Quality Testing", value=True)
run_advanced = st.sidebar.checkbox("Run Advanced Analytics", value=True)
run_dashboards = st.sidebar.checkbox("Generate Dashboards", value=True)

if uploaded_file:
    # --- Load Data ---
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
    elif file_type == 'json':
        df = pd.read_json(uploaded_file)
    elif file_type == 'parquet':
        df = pd.read_parquet(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.success(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(100))

    # --- Data Cleaning ---
    if run_cleaning:
        st.subheader("🧹 Data Cleaning")
        cleaner = DataCleaner()
        cleaned_data, cleaning_stats = cleaner.comprehensive_clean(df)
        st.write("**Cleaning Summary:**", cleaner.get_cleaning_summary())
        st.dataframe(cleaned_data.head(100))
    else:
        cleaned_data = df.copy()

    # --- Data Quality Testing ---
    if run_testing:
        st.subheader("🧪 Data Quality Testing")
        tester = DataTestingUtils()
        test_report = tester.generate_test_report(cleaned_data)
        st.write("**Test Report Summary:**", test_report['summary'])
        if test_report['summary']['recommendations']:
            st.warning("\n".join(test_report['summary']['recommendations']))

    # --- Analytics & Visualizations ---
    st.subheader("📊 Analytics & Visualizations")
    model = DataAnalyticsModel(cleaned_data)
    quality_report = model.analyze_data_quality()
    st.write("**Quality Report:**", quality_report)

    # --- Advanced Analytics ---
    if run_advanced:
        st.subheader("🚀 Advanced Analytics")
        adv = AdvancedAnalytics()
        adv_results = adv.run_comprehensive_advanced_analysis(cleaned_data)
        st.write("**Anomaly Detection:**", adv_results.get('anomaly_detection', {}))
        st.write("**Clustering:**", adv_results.get('clustering', {}))
        st.write("**PCA Analysis:**", adv_results.get('pca_analysis', {}))
        if 'temporal_analysis' in adv_results:
            st.write("**Temporal Analysis:**", adv_results['temporal_analysis'])

    # --- Dashboards ---
    if run_dashboards:
        st.subheader("📈 Dashboards")
        dash = DashboardGenerator()
        dashboards = dash.generate_all_dashboards(cleaned_data, {'summary': quality_report})
        st.success("Dashboards generated! Open the HTML files in the dashboard_output folder.")
        for name, path in dashboards.items():
            st.write(f"[{name.capitalize()} Dashboard]({path})")

    # --- Download Cleaned Data ---
    st.subheader("⬇️ Download Cleaned Data")
    csv = cleaned_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.info("Upload a data file to get started.") 