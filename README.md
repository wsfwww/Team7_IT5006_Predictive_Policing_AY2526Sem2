# Chicago Crime Spatio-Temporal Analysis

### IT5006 Project Group 7

Group Member: HUANG YAOXUAN, QI JINGYANG, XIANG JIAJIA, ZHANG ZIRUI, ZHOU YAYUE

## 🚀 Live Dashboard Demo

Access the interactive dashboard here: **[Chicago Crime Dashboard](https://it5006projectgroup7-aswjrshvzdti2wnkngxbks.streamlit.app/)**

---

## 📂 Repository Structure & File Descriptions

| File / Folder | Description |
| --- | --- |
| **`Team7_Phase1_IT5006_EDA.ipynb`** | **EDA Core Code** Include Temporal pattern analysis, Spatial distribution study, Crime correlation analysis, Key insights and patterns discovered. |
| **`dashboard.py`** | **The Core Application.** The main Streamlit script that handles the UI layout, interactive filtering, and visualization logic using Plotly and Matplotlib. |
| **`gen_data_chunk.ipynb`** | **Data Pipeline.** The Jupyter Notebook used for data slicing and preprocessing. It cleans the raw Chicago Crime dataset, optimizes memory usage (e.g., using `category` types), and exports data into Parquet chunks. |
| **`data_chunks/`** | **Processed Data.** Contains 11 Parquet files (~170MB total). This chunked storage approach ensures efficient memory management and faster loading times on Streamlit Cloud. |
| **`requirements.txt`** | **Environment Setup.** List of required Python libraries (Streamlit, Pandas, Plotly, Matplotlib, Pyarrow, etc.) for local and cloud deployment. |

---
