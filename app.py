
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="💧 Smart Water Management Dashboard",
    page_icon="🌿",
    layout="wide"
)

# ==========================
# LOAD / PREPARE DATA
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("TARP.csv")

    # Add synthetic columns if missing
    if "Climate" not in df.columns:
        df["Climate"] = np.random.choice(["Sunny", "Rainy", "Cloudy", "Humid"], len(df))
    if "Crop Type" not in df.columns:
        df["Crop Type"] = np.random.choice(["Rice", "Wheat", "Maize", "Sugarcane"], len(df))
    if "Soil Type" not in df.columns:
        df["Soil Type"] = np.random.choice(["Sandy", "Clay", "Loamy"], len(df))
    if "Land Size" not in df.columns:
        df["Land Size"] = np.random.randint(1, 10, len(df))
    if "Soil Humidity" not in df.columns:
        df["Soil Humidity"] = np.random.uniform(20, 90, len(df))

    if "Water Need (%)" not in df.columns:
        df["Water Need (%)"] = np.random.uniform(30, 90, len(df))
    if "Irrigation Time (min)" not in df.columns:
        df["Irrigation Time (min)"] = np.random.uniform(20, 100, len(df))

    return df

df = load_data()

# ==========================
# MODEL TRAINING
# ==========================
def train_model(df):
    label_encoders = {}
    for col in ["Climate", "Crop Type", "Soil Type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df[["Soil Humidity", "Climate", "Crop Type", "Soil Type", "Land Size"]]
    y1 = df["Water Need (%)"]
    y2 = df["Irrigation Time (min)"]

    model_water = RandomForestRegressor(random_state=42)
    model_irrigation = RandomForestRegressor(random_state=42)

    model_water.fit(X, y1)
    model_irrigation.fit(X, y2)

    return model_water, model_irrigation, label_encoders

model_water, model_irrigation, encoders = train_model(df)

# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.title("💧 Smart Water Management System")
menu = st.sidebar.radio("Select Page", ["📘 Info", "🌾 Prediction", "📊 Visualization"])

# ==========================
# PAGE 1: INFO
# ==========================
if menu == "📘 Info":
    st.title("💧 Smart Water Management System for Agriculture Fields 🌿")

    st.markdown("""
    ### 🌱 Project Overview
    This system uses AI and machine learning to **predict water requirements** 
    and **irrigation durations** for crops based on various agricultural factors.

    ### ⚙️ Features
    - Predicts **Water Need (%)** and **Irrigation Time (min)**
    - Based on **Soil Humidity**, **Climate**, **Crop Type**, **Soil Type**, and **Land Size**
    - Provides rich **data visualizations** for better understanding
    - Built using **Python, Streamlit, and Random Forest Regression**

    ### 🎯 Objective
    To help farmers **reduce water wastage** and **improve irrigation efficiency** using smart analytics.
    """)

# ==========================
# PAGE 2: PREDICTION
# ==========================
elif menu == "🌾 Prediction":
    st.title("🌾 Water Requirement Prediction Dashboard")

    st.markdown("### 🔧 Input Field Parameters")

    col1, col2 = st.columns(2)

    with col1:
        soil_humidity = st.slider("Soil Humidity (%)", 0.0, 100.0, 50.0)
        climate = st.selectbox("Climate", encoders["Climate"].classes_)
        crop = st.selectbox("Crop Type", encoders["Crop Type"].classes_)

    with col2:
        soil_type = st.selectbox("Soil Type", encoders["Soil Type"].classes_)
        land_size = st.slider("Land Size (Acres)", 0.5, 20.0, 5.0)

    st.markdown("---")

    if st.button("🔍 Predict Water Requirement"):
        # Encode input values
        climate_encoded = encoders["Climate"].transform([climate])[0]
        crop_encoded = encoders["Crop Type"].transform([crop])[0]
        soil_encoded = encoders["Soil Type"].transform([soil_type])[0]

        input_data = np.array([[soil_humidity, climate_encoded, crop_encoded, soil_encoded, land_size]])

        water_pred = model_water.predict(input_data)[0]
        time_pred = model_irrigation.predict(input_data)[0]

        st.success(f"💧 **Predicted Water Need:** {water_pred:.2f}%")
        st.success(f"⏱️ **Estimated Irrigation Time:** {time_pred:.2f} minutes")

        st.markdown("### 🧾 Prediction Summary")
        result_df = pd.DataFrame({
            "Soil Humidity (%)": [soil_humidity],
            "Climate": [climate],
            "Crop Type": [crop],
            "Soil Type": [soil_type],
            "Land Size (Acres)": [land_size],
            "Predicted Water Need (%)": [f"{water_pred:.2f}"],
            "Irrigation Time (min)": [f"{time_pred:.2f}"]
        })
        st.dataframe(result_df)

# ==========================
# PAGE 3: VISUALIZATION
# ==========================
elif menu == "📊 Visualization":
    st.title("📊 Smart Agriculture Data Visualization")

    st.markdown("### Explore patterns between various features and irrigation requirements")

    st.subheader("1️⃣ Soil Humidity vs Water Need")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Soil Humidity", y="Water Need (%)", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("2️⃣ Climate-wise Water Need")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Climate", y="Water Need (%)", data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("3️⃣ Crop Type vs Irrigation Time")
    fig3, ax3 = plt.subplots()
    sns.barplot(x="Crop Type", y="Irrigation Time (min)", data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("4️⃣ Soil Type vs Water Need")
    fig4, ax4 = plt.subplots()
    sns.violinplot(x="Soil Type", y="Water Need (%)", data=df, ax=ax4)
    st.pyplot(fig4)

    st.subheader("5️⃣ Land Size vs Irrigation Time")
    fig5, ax5 = plt.subplots()
    sns.lineplot(x="Land Size", y="Irrigation Time (min)", data=df, ax=ax5, marker="o")
    st.pyplot(fig5)

    st.markdown("✅ **All charts are generated dynamically using real or synthetic agricultural data.**")
