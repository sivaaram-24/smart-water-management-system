
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

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
    df = pd.DataFrame({
        "Climate": np.random.choice(["Sunny", "Rainy", "Cloudy", "Humid"], 100),
        "Crop Type": np.random.choice(["Rice", "Wheat", "Maize", "Sugarcane"], 100),
        "Soil Type": np.random.choice(["Sandy", "Clay", "Loamy"], 100),
        "Land Size": np.random.randint(1, 10, 100),
        "Soil Humidity": np.random.uniform(20, 90, 100),
        "Water Need (%)": np.random.uniform(30, 90, 100),
        "Irrigation Time (min)": np.random.uniform(20, 100, 100)
    })
    return df

df = load_data()

# ==========================
# TRAIN MODELS
# ==========================
def train_models(df):
    label_encoders = {}
    for col in ["Climate", "Crop Type", "Soil Type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[["Soil Humidity", "Climate", "Crop Type", "Soil Type", "Land Size"]]
    y1 = df["Water Need (%)"]
    y2 = df["Irrigation Time (min)"]

    model1 = RandomForestRegressor(random_state=42)
    model2 = RandomForestRegressor(random_state=42)
    model1.fit(X, y1)
    model2.fit(X, y2)
    return model1, model2, label_encoders

model_water, model_irrigation, encoders = train_models(df)

# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.title("💧 Smart Water Management System")
menu = st.sidebar.radio("Navigate", ["📘 Info", "🌾 Prediction", "📊 Visualization"])

# ==========================
# PAGE 1: INFO
# ==========================
if menu == "📘 Info":
    st.title("💧 Smart Water Management System for Agriculture Fields 🌿")
    st.markdown("""
    ### 🌱 Project Overview
    This system helps predict **water requirements** and **irrigation duration**
    for different crops under varying environmental conditions.

    ### ⚙️ Features
    - Predicts **Water Need (%)** and **Irrigation Time (min)**
    - Inputs: **Soil Humidity**, **Climate**, **Crop Type**, **Soil Type**, **Land Size**
    - Interactive data visualization and analytics
    - Built with **Python, Streamlit, and Machine Learning**

    ### 🎯 Goal
    To promote **efficient water usage** and **reduce irrigation waste** through smart predictions.
    """)

# ==========================
# PAGE 2: PREDICTION
# ==========================
elif menu == "🌾 Prediction":
    st.title("🌾 Predict Water Requirement and Irrigation Time")

    st.markdown("### 🔧 Input Parameters")

    col1, col2 = st.columns(2)
    with col1:
        soil_humidity = st.slider("Soil Humidity (%)", 0.0, 100.0, 50.0)
        climate = st.selectbox("Climate", encoders["Climate"].classes_)
        crop = st.selectbox("Crop Type", encoders["Crop Type"].classes_)

    with col2:
        soil_type = st.selectbox("Soil Type", encoders["Soil Type"].classes_)
        land_size = st.slider("Land Size (Acres)", 0.5, 20.0, 5.0)

    if st.button("🔍 Predict"):
        # Encode inputs
        climate_encoded = encoders["Climate"].transform([climate])[0]
        crop_encoded = encoders["Crop Type"].transform([crop])[0]
        soil_encoded = encoders["Soil Type"].transform([soil_type])[0]

        X_input = np.array([[soil_humidity, climate_encoded, crop_encoded, soil_encoded, land_size]])
        water_need = model_water.predict(X_input)[0]
        irrigation_time = model_irrigation.predict(X_input)[0]

        st.success(f"💧 **Predicted Water Need:** {water_need:.2f}%")
        st.success(f"⏱️ **Estimated Irrigation Time:** {irrigation_time:.2f} minutes")

        # Display results table
        st.markdown("### 🧾 Prediction Summary")
        result_df = pd.DataFrame({
            "Soil Humidity (%)": [soil_humidity],
            "Climate": [climate],
            "Crop Type": [crop],
            "Soil Type": [soil_type],
            "Land Size (Acres)": [land_size],
            "Predicted Water Need (%)": [f"{water_need:.2f}"],
            "Irrigation Time (min)": [f"{irrigation_time:.2f}"]
        })
        st.dataframe(result_df)

# ==========================
# PAGE 3: VISUALIZATION
# ==========================
elif menu == "📊 Visualization":
    st.title("📊 Smart Agriculture Visualization")

    st.markdown("### Explore relationships between environmental and irrigation features")

    # 1️⃣ Soil Humidity vs Water Need
    st.subheader("1️⃣ Soil Humidity vs Water Need")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Soil Humidity", y="Water Need (%)", hue="Climate", data=df, ax=ax1)
    st.pyplot(fig1)

    # 2️⃣ Crop Type vs Irrigation Time
    st.subheader("2️⃣ Crop Type vs Irrigation Time")
    fig2, ax2 = plt.subplots()
    sns.barplot(x="Crop Type", y="Irrigation Time (min)", data=df, ax=ax2)
    st.pyplot(fig2)

    # 3️⃣ Soil Type vs Water Need
    st.subheader("3️⃣ Soil Type vs Water Need")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Soil Type", y="Water Need (%)", data=df, ax=ax3)
    st.pyplot(fig3)

    # 4️⃣ Land Size vs Irrigation Time
    st.subheader("4️⃣ Land Size vs Irrigation Time")
    fig4, ax4 = plt.subplots()
    sns.lineplot(x="Land Size", y="Irrigation Time (min)", data=df, marker="o", ax=ax4)
    st.pyplot(fig4)

    # Filter section
    st.markdown("---")
    st.subheader("🔍 Filter Data by Crop and Climate")
    selected_crop = st.selectbox("Select Crop Type:", df["Crop Type"].unique())
    selected_climate = st.selectbox("Select Climate:", df["Climate"].unique())
    filtered_df = df[(df["Crop Type"] == selected_crop) & (df["Climate"] == selected_climate)]
    st.dataframe(filtered_df.head(10))

    st.info("✅ Interactive charts help analyze how each factor affects water usage.")
