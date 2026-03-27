%%writefile app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load files
model = pickle.load(open("/content/drive/MyDrive/agri_project/final_model.pkl", "rb"))
scaler = pickle.load(open("/content/drive/MyDrive/agri_project/scaler.pkl", "rb"))
encoders = pickle.load(open("/content/drive/MyDrive/agri_project/encoders.pkl", "rb"))
features = pickle.load(open("/content/drive/MyDrive/agri_project/feature_columns.pkl", "rb"))

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Recommendation
def get_recommendation(pred):
    if pred < 200:
        return "⚠️ Low Yield → Improve irrigation & fertilizer"
    elif pred < 500:
        return "⚡ Moderate Yield → Optimize nutrients"
    else:
        return "✅ High Yield → Maintain current practices"

# Page config
st.set_page_config(page_title="AgriYield Dashboard", layout="wide")

st.markdown("<h1 style='text-align:center; color:green;'>🌾 AI AgriYield Predictor</h1>", unsafe_allow_html=True)

# Dropdown values
state_list = list(encoders["state_name"].classes_)
district_list = list(encoders["dist_name"].classes_)
crop_list = list(encoders["crop"].classes_)
soil_list = list(encoders["soil_type"].classes_)

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", value=2015)
    state = st.selectbox("State", state_list)
    district = st.selectbox("District", district_list)
    crop = st.selectbox("Crop", crop_list)
    soil = st.selectbox("Soil Type", soil_list)

with col2:
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 2000.0, 1200.0)
    wind = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
    solar = st.slider("Solar Radiation (MJ/m²/day)", 0.0, 30.0, 18.0)

with col3:
    ph = st.slider("pH", 0.0, 14.0, 6.5)
    
    # ✅ Don’t know NPK checkbox
    unknown_npk = st.checkbox("I don't know N, P, K / total nutrients (Auto-fill)")
    
    if unknown_npk:
        # Fill default or average values (can customize based on crop/soil)
        n = 10.0
        p = 5.0
        tn = 4000000.0
        tp = 2200000.0
        tk = 4000000.0
        st.info("Using default NPK values for prediction")
    else:
        n = st.number_input("Nitrogen (kg/ha)", value=8.4)
        p = st.number_input("Phosphorus (kg/ha)", value=4.0)
        tn = st.number_input("Total N", value=4624983.0)
        tp = st.number_input("Total P", value=2219991.0)
        tk = st.number_input("Total K", value=4069985.0)

# Predict
if st.button("🌾 Predict Yield"):
    input_dict = {
        "year": year,
        "state_name": state,
        "dist_name": district,
        "crop": crop,
        "soil_type": soil,
        "temperature_c": temp,
        "humidity_%": humidity,
        "rainfall_mm": rainfall,
        "wind_speed_m_s": wind,
        "solar_radiation_mj_m2_day": solar,
        "ph": ph,
        "n_req_kg_per_ha": n,
        "p_req_kg_per_ha": p,
        "total_n_kg": tn,
        "total_p_kg": tp,
        "total_k_kg": tk
    }

    df = pd.DataFrame([input_dict])

    # Encode
    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    base_features = [
        'year', 'state_name', 'dist_name', 'crop', 'soil_type',
        'temperature_c', 'humidity_%', 'rainfall_mm',
        'wind_speed_m_s', 'solar_radiation_mj_m2_day',
        'ph', 'n_req_kg_per_ha', 'p_req_kg_per_ha',
        'total_n_kg', 'total_p_kg', 'total_k_kg'
    ]

    df_base = df[base_features]

    # Scale
    df_scaled = scaler.transform(df_base)
    df_scaled = pd.DataFrame(df_scaled, columns=base_features)

    # Feature Engineering
    df_scaled["temp_rain_interaction"] = df["temperature_c"] * df["rainfall_mm"]
    df_scaled["humidity_temp_index"] = df["humidity_%"] * df["temperature_c"]

    df_final = df_scaled[features]

    prediction = model.predict(df_final)[0]
    recommendation = get_recommendation(prediction)

    # Save history
    history_entry = {
        "State": state,
        "District": district,
        "Crop": crop,
        "Yield": round(prediction, 2),
        "NPK Known": not unknown_npk
    }
    st.session_state.history.append(history_entry)

    # Output
    st.success(f"🌾 Predicted Yield: {prediction:.2f} kg/ha")
    st.info(f"📊 Recommendation: {recommendation}")

    # Environmental chart
    st.subheader("📊 Environmental Impact")
    fig, ax = plt.subplots()
    labels = ["Temperature", "Rainfall", "Humidity"]
    values = [temp, rainfall, humidity]
    ax.bar(labels, values, color=["orange", "blue", "green"])
    ax.set_title("Environmental Factors")
    st.pyplot(fig)

    # Report
    report = f"""
    AGRIYIELD REPORT

    Yield: {prediction:.2f} kg/ha

    State: {state}
    District: {district}
    Crop: {crop}

    Recommendation:
    {recommendation}

    NPK values used: {'Default/Estimated' if unknown_npk else 'Farmer Provided'}
    """
    st.download_button("📄 Download Report", report, file_name="agri_report.txt")

# HISTORY SECTION
st.subheader("📜 Prediction History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History", csv, "history.csv")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History Cleared!")

    # Yield trend chart
    st.subheader("📈 Yield Trend")
    history_df["Index"] = range(1, len(history_df) + 1)
    fig2, ax2 = plt.subplots()
    ax2.plot(history_df["Index"], history_df["Yield"], marker='o', color='green')
    ax2.fill_between(history_df["Index"], history_df["Yield"], color='lightgreen', alpha=0.5)
    ax2.set_xlabel("Prediction Number")
    ax2.set_ylabel("Yield")
    ax2.set_title("Prediction Trend")
    st.pyplot(fig2)
else:
    st.write("No history yet")
