import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# ✅ Set page config at the top
#st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")

# ================================
# 📂 Load Data
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv('DATA/Crop_production.csv')
    return df

df = load_data()
df1 = df.copy()

# Drop unnecessary columns
df1.drop(['Unnamed: 0', 'State_Name', 'Crop_Type', 'Crop'], axis=1, inplace=True)

# Label Encoding
state_encoder = LabelEncoder()
crop_type_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

state_encoder.fit(df['State_Name'])
crop_type_encoder.fit(df['Crop_Type'])
crop_encoder.fit(df['Crop'])

# Add encoded categorical features
df1['State_Name_Encoded'] = state_encoder.transform(df['State_Name'])
df1['Crop_Type_Encoded'] = crop_type_encoder.transform(df['Crop_Type'])
df1['Crop_Encoded'] = crop_encoder.transform(df['Crop'])

# Features & Target Variable
X = df1.drop('Yield_ton_per_hec', axis=1)
y = df1['Yield_ton_per_hec']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle NaN values
y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(y_train.mean())
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())

# Train XGBoost Model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# ================================
# 🤖 Gemini API Setup
# ================================
genai.configure(api_key="API_KEY")  # Replace with your Gemini API Key 

def get_best_crop_suggestion(N, P, K, temperature, pH, area, rainfall, production, state):
    prompt = f"""
    Given the following soil and climate conditions:
    - Nitrogen: {N}
    - Phosphorus: {P}
    - Potassium: {K}
    - Temperature: {temperature}°C
    - pH: {pH}
    - Area: {area} hectares
    - Rainfall: {rainfall} mm
    - Production: {production} tons
    - Current State: {state}

    The user is not satisfied with the predicted yield.
    Suggest which crop they should grow and in which Indian state to get maximum yield.
    """

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text

# ================================
# 🎨 Streamlit UI
# ================================

# Sidebar Navigation
st.sidebar.header("🌿 Crop Yield Prediction")

# Sidebar: Yield Satisfaction Button
st.sidebar.subheader("🤔 Are you satisfied with the yield?")
satisfaction_clicked = st.sidebar.button("🔍 Get best Crop Yield Suggestion")

# ================================
# 📝 User Inputs
# ================================

st.markdown("<h1 style='text-align: center; color: #008000;'> 🌾 Crop Yield Prediction System</h1>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌱 Nitrogen (N) content", min_value=0.0, step=0.1)
    P = st.number_input("🌾 Phosphorus (P) content", min_value=0.0, step=0.1)
    K = st.number_input("🍀 Potassium (K) content", min_value=0.0, step=0.1)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, step=0.1)
    
with col2:
    pH = st.number_input("🧪 Soil pH Level", min_value=0.0, step=0.1)
    area = st.number_input("📏 Area in hectares", min_value=0.1, step=0.1)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=1.0)
    production = st.number_input("🏭 Production (tons)", min_value=0.0, step=0.1)

# Select State, Crop Type, Crop
state_selected = st.selectbox("🏛️ Select State", state_encoder.classes_)
crop_type_selected = st.selectbox("🌿 Select Crop Type", crop_type_encoder.classes_)
crop_selected = st.selectbox("🌾 Select Crop", crop_encoder.classes_)

# Predict Button
if st.button("🚀 Predict Yield"):
    state_encoded = state_encoder.transform([state_selected])[0]
    crop_type_encoded = crop_type_encoder.transform([crop_type_selected])[0]
    crop_encoded = crop_encoder.transform([crop_selected])[0]

    # Create Input DataFrame
    input_data = pd.DataFrame([{
        'N': N, 'P': P, 'K': K, 'pH': pH, 'rainfall': rainfall, 'temperature': temperature,
        'Area_in_hectares': area, 'Production_in_tons': production,
        'State_Name_Encoded': state_encoded, 'Crop_Type_Encoded': crop_type_encoded, 'Crop_Encoded': crop_encoded
    }])

    # Ensure same column order
    input_data = input_data[X.columns]

    # Predict Yield
    predicted_yield = model.predict(input_data)[0] * 907.185  # Convert to Kg
    st.success(f"🎯 **Predicted Yield:** {predicted_yield:.2f} Kg ✅")

# ================================
# 📌 Gemini AI Suggestion (Centered)
# ================================
if satisfaction_clicked:
    st.subheader("🔍 **Best Crop Suggestion**")
    suggestion = get_best_crop_suggestion(N, P, K, temperature, pH, area, rainfall, production, state_selected)

    st.info(f"🌟 **AI Suggestion:** {suggestion}")
