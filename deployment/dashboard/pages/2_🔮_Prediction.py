import streamlit as st
import requests
import json

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Crime Arrest Predictor",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Police Arrest Rate Prediction System")
st.markdown("""
This dashboard predicts the likelihood of an arrest being made for a reported crime 
based on historical data and machine learning models. Please input the incident details below.
""")

st.divider()

# ==========================================
# 2. Form Inputs (UI Layout)
# ==========================================
st.subheader("Incident Details")

# Define valid options based on model features
VALID_LOCATIONS = [
    "ATM Separate from Bank", "Abandoned/Condemned Structure", "Air/Bus/Train Terminal",
    "Amusement Park", "Arena/Stadium/Fairgrounds/Coliseum", "Auto Dealership New/Used",
    "Bank/Savings and Loan", "Bar/Nightclub", "Camp/Campground",
    "Church/Synagogue/Temple/Mosque", "Commercial/Office Building", "Community Center",
    "Construction Site", "Convenience Store", "Cyberspace", "Daycare Facility",
    "Department/Discount Store", "Dock/Wharf/Freight/Modal Terminal",
    "Drug Store/Doctor's Office/Hospital", "Farm Facility", "Field/Woods",
    "Gambling Facility/Casino/Race Track", "Government/Public Building",
    "Grocery/Supermarket", "Highway/Road/Alley/Street/Sidewalk", "Hotel/Motel/Etc.",
    "Industrial Site", "Jail/Prison/Penitentiary/Corrections Facility",
    "Lake/Waterway/Beach", "Liquor Store", "Military Installation", "Other/Unknown",
    "Park/Playground", "Parking/Drop Lot/Garage", "Rental Storage Facility",
    "Residence/Home", "Rest Area", "Restaurant", "School-College/University",
    "School-Elementary/Secondary", "School/College", "Service/Gas Station",
    "Shelter-Mission/Homeless", "Shopping Mall", "Specialty Store", "Tribal Lands"
]

VALID_OFFENSES = [
    "Animal Cruelty", "Arson", "Assault Offenses", "Bribery",
    "Burglary/Breaking & Entering", "Counterfeiting/Forgery",
    "Destruction/Damage/Vandalism of Property", "Drug/Narcotic Offenses",
    "Embezzlement", "Extortion/Blackmail", "Fraud Offenses",
    "Gambling Offenses", "Homicide Offenses", "Human Trafficking",
    "Kidnapping/Abduction", "Larceny/Theft Offenses", "Motor Vehicle Theft",
    "Other/Unknown", "Pornography/Obscene Material", "Prostitution Offenses",
    "Robbery", "Sex Offenses", "Sex Offenses, Non-forcible",
    "Stolen Property Offenses", "Weapon Law Violations"
]

DAYS_OF_WEEK = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

st.subheader("Model Selection")
model_choice = st.radio(
    "Choose Machine Learning Model:",
    options=["LightGBM (High Accuracy)", "Logistic Regression (Baseline)"],
    horizontal=True
)

model_name_map = {
    "LightGBM (High Accuracy)": "lgbm",
    "Logistic Regression (Baseline)": "lr"
}
selected_model_key = model_name_map[model_choice]

st.divider()

st.subheader("Incident Details")
# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Time & Date**")
    hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=12)
    month = st.selectbox("Month", options=list(range(1, 13)), index=6)
    day_string = st.selectbox("Day of Week", options=list(DAYS_OF_WEEK.keys()))
    is_holiday_str = st.radio("Is it a Public Holiday?", options=["No", "Yes"])

with col2:
    st.markdown("**Location & Target**")
    location = st.selectbox("Location Type", options=VALID_LOCATIONS)
    crime_against = st.selectbox("Crime Against", options=["Person", "Property", "Society", "Other"])

with col3:
    st.markdown("**Offense Category**")
    offense_category = st.selectbox("Offense Category", options=VALID_OFFENSES)

st.divider()

# ==========================================
# 3. Prediction Logic
# ==========================================
# API endpoint (Change this to your Render URL when deploying the Dashboard!)
# Example: API_URL = "https://your-flask-api.onrender.com/predict"
# API_URL="http://localhost:5001/predict"
API_URL = "https://prediction-api-vlwb.onrender.com/predict"

if st.button("🔍 Predict Arrest Probability", type="primary", use_container_width=True):
    
    # Map user inputs to backend JSON format
    payload = {
        "model_name": selected_model_key,
        "hour": hour,
        "month": month,
        "weekday": DAYS_OF_WEEK[day_string],
        "is_holiday": 1 if is_holiday_str == "Yes" else 0,
        "location_name": location,
        "crime_against": crime_against,
        "offense_category_name": offense_category
    }
    
    with st.spinner("Analyzing crime patterns... (Note: First prediction may take up to 2 minutes to wake up the server)"):
        try:
            # Send POST request to the Flask API
            response = requests.post(API_URL, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    prob = result.get("arrest_probability", 0.0)
                    prediction_label = result.get("prediction_label", 0)
                    
                    # Display results in a visually appealing way
                    st.subheader("Prediction Results")
                    
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        if prediction_label == 1:
                            st.metric(label="Arrest Outcome", value="Likely Arrested")
                        else:
                            st.metric(label="Arrest Outcome", value="Unlikely Arrested")
                            
                    with res_col2:
                        st.markdown("**Probability of Arrest**")
                        st.progress(prob)
                        st.write(f"The model predicts a **{prob:.1%}** chance of the suspect being arrested.")
                        
                        if prob > 0.5:
                            st.info("💡 Tip: Crimes with this profile historically show a strong police clearance rate.")
                        else:
                            st.warning("💡 Tip: Crimes with this profile historically have lower arrest rates. More evidence might be required.")
                            
                else:
                    st.error(f"Backend Error: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"Failed to connect to API. Status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Failed! Please make sure your backend Flask API is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")