# Chicago Crime Spatio-Temporal Analysis

### IT5006 Project Group 7

Group Member: HUANG YAOXUAN, QI JINGYANG, XIANG JIAJIA, ZHANG ZIRUI, ZHOU YAYUE

## 🚀 Live Dashboard Demo

Access the interactive dashboard here: **[Chicago Crime Dashboard](https://it5006projectgroup7-aswjrshvzdti2wnkngxbks.streamlit.app/)**

---

## 📂 Repository Structure & File Descriptions

| Component | Location | Description |
|-----------|----------|-------------|
| **EDA Notebook** | `Team7_Phase1_IT5006_EDA.ipynb` | Comprehensive exploratory analysis including temporal patterns, spatial distribution, crime correlations, and key insights |
| **Model Notebook** | `Team7_Phase2_IT5006_AY2526Sem2.ipynb` | Model training and evaluation with Logistic Regression, Deep Learning (MLP), Random Forest, and LightGBM |
| **Data Processing** | `gen_data_chunk.ipynb` | Data preprocessing pipeline that cleans and chunks Chicago crime dataset for optimal memory usage |
| **Dashboard** | `deployment/eda_dashboard/dashboard.py` | Main Streamlit application for interactive visualization, filtering, and real-time exploration |
| **Data Chunks** | `deployment/eda_dashboard/data_chunks/` | Processed crime data in Parquet format optimized for cloud deployment |
| **Model API** | `deployment/model_api/` | Flask-based API server for model inference and predictions |
| **Dependencies** | `requirements.txt` | Complete list of Python packages and versions |

---
