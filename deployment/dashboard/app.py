import streamlit as st

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Crime Arrest Predictor",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Main Content Area
st.title("🚓 Crime Arrest Prediction System")
st.markdown("---")

st.markdown("""
### Welcome to our Project Dashboard!

This application is designed to analyze historical crime data and predict the likelihood of an arrest for incoming incidents. Our models are engineered to capture generalized policing logic, allowing for robust inferences across different datasets and jurisdictions.

**Please select a module from the sidebar to begin:**

* **📊 EDA Dashboard:** Explore historical crime trends, spatial hotspots, and temporal distributions specifically across **Chicago** neighborhoods.
* **🔮 Prediction API:** Input incident details (time, location, crime type) to get a real-time arrest probability prediction. You can choose to run inferences using either our high-accuracy **LightGBM** model or our baseline **Logistic Regression** model, both seamlessly integrated into our cloud-hosted microservice.
""")
