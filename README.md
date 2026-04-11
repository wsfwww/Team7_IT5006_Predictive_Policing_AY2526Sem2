# Chicago Crime Spatio-Temporal Analysis

## IT5006 Project Group 7

Group Member: HUANG YAOXUAN, QI JINGYANG, XIANG JIAJIA, ZHANG ZIRUI, ZHOU YAYUE

## Live Dashboard Demo

Access the interactive dashboard here: **[Chicago Crime Dashboard](https://it5006projectgroup7-aswjrshvzdti2wnkngxbks.streamlit.app/)**

---

## Repository Structure & File Descriptions

| Component | Location | Description |
|-----------|----------|-------------|
| **EDA Notebook** | `src/Team7_Phase1_IT5006_EDA.ipynb` | Comprehensive exploratory analysis including temporal patterns, spatial distribution, crime correlations, and key insights |
| **Model Notebook** | `src/Team7_Phase2_IT5006_AY2526Sem2.ipynb` | Model training and evaluation with Logistic Regression, Deep Learning (MLP), Random Forest, and LightGBM |
| **Data Processing** | `notebooks/gen_data_chunk.ipynb` | Data preprocessing pipeline that cleans and chunks Chicago crime dataset for optimal memory usage |
| **Dashboard** | `deployment/eda_dashboard/dashboard.py` | Main Streamlit application for interactive visualization, filtering, and real-time exploration |
| **Data Chunks** | `deployment/eda_dashboard/data_chunks/` | Processed crime data in Parquet format optimized for cloud deployment |
| **Model API** | `deployment/model_api/` | Flask-based API server for model inference and predictions |
| **Dependencies** | `requirements.txt` | Complete list of Python packages and versions |

---
## Data

Raw data: https://drive.google.com/file/d/1SYksz0ACxpGQjxSjSdEh23oR_AjBgWOJ/view?usp=drive_link

Processed data for models: https://drive.google.com/drive/folders/1h6-y723YKJ1SbAZFfH_ohzG80N9YP2Zc?usp=drive_link

---

## Setup Guide

This guide will help you quickly configure the environment and run this project.

---

### System Requirements

- **Python Version**: 3.8 or higher (recommended 3.10+)
- **Operating System**: Windows / macOS / Linux
- **Disk Space**: At least 5GB (including all dependencies and data)
- **RAM**: At least 8GB (recommended 16GB)

### Check Python Version
```bash
python --version
# Should output Python 3.8+ or Python 3.10+
```

---

### Installation Methods

### Method 1: Using pip (Recommended)

#### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

#### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Verify Installation
```bash
python -c "import streamlit, pandas, torch; print('Installation successful!')"
```

---

### Method 2: Using Conda (Recommended for PyTorch)

#### Step 1: Create Conda Environment
```bash
conda create -n crime-prediction python=3.10
conda activate crime-prediction
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: GPU Support (NVIDIA Graphics Card)
```bash
# Uninstall CPU version PyTorch
pip uninstall torch torchvision -y

# Install GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Running the Project

### 1. Start Streamlit Dashboard
```bash
streamlit run dashboard.py
```

Output should display:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Then open http://localhost:8501 in your browser.

### 2. Run Jupyter Notebooks
```bash
jupyter notebook

# Or use JupyterLab (more modern)
jupyter lab
```

Then open the following notebooks:
- `Team7_Phase1_IT5006_EDA.ipynb` - Exploratory Data Analysis
- `Team7_Phase2_IT5006_AY2526Sem2.ipynb` - Model Training & Evaluation

### 3. Run Data Processing Script
```bash
jupyter notebook gen_data_chunk.ipynb
```

---

### Configuration

### Data Path Configuration

Modify the data path in `gen_data_chunk.ipynb`:
```python
base_path = '/path/to/your/data'  # Change to your data path
path_chicago = f'{base_path}/Chicago'
path_nibrs = f'{base_path}/NIBRS/CA-2024'
```

### Streamlit Configuration

Optional: Customize in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[server]
maxUploadSize = 200
```

---

### Troubleshooting

### Issue 1: Slow pip Installation
**Solution**: Use a faster mirror source
```bash
pip install -r requirements.txt -i https://pypi.org/simple/
```

### Issue 2: PyTorch Installation Fails
**Solution**:
```bash
# Method 1: Use official source
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Method 2: Use conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Issue 3: Out of Memory Error
**Solution**:
- Close other programs to free up memory
- Use GPU acceleration (if you have an NVIDIA graphics card)
- Load data in batches

### Issue 4: Streamlit Cannot Find Data Files
**Solution**:
- Check if `data_chunks/` folder exists
- Verify file paths do not contain Chinese or special characters
- Try using absolute paths

### Issue 5: Google Colab Environment
**Solution**: Notebook code already has Google Colab support
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### Virtual Environment Management

### View Installed Packages
```bash
pip list
```

### Update Specific Package
```bash
pip install --upgrade pandas
```

### Export Current Environment
```bash
pip freeze > requirements.txt
```

### Delete Virtual Environment
```bash
# Windows
rmdir venv /s

# macOS / Linux
rm -rf venv
```

---

## Verify Installation Completion

Run the following command to verify all critical dependencies:
```bash
python -c "
import streamlit
import pandas as pd
import numpy as np
import torch
import lightgbm
import sklearn
print('All dependencies installed successfully!')
print(f'Streamlit: {streamlit.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'LightGBM: {lightgbm.__version__}')
"
```


## Quick Start Example

```bash
# 1. Clone repository
git clone https://github.com/wsfwww/Team7_IT5006_Predictive_Policing_AY2526Sem2.git
cd Team7_IT5006_Predictive_Policing_AY2526Sem2

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run dashboard
streamlit run dashboard.py

# 5. Open browser and navigate to http://localhost:8501
```
