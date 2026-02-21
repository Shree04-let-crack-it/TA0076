import streamlit as st
import base64

# ==============================
# 🌾 Page Configuration
# ==============================
st.set_page_config(
    page_title="Smart Farming AI",
    page_icon="🌾",
    layout="wide"
)

# ==============================
# 🎨 Background Image Function
# ==============================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    page_bg = f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
            url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    h1, h2, h3, p {{
        color: white;
    }}

    .stButton>button {{
        height: 70px;
        font-size: 18px;
        border-radius: 12px;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# 🔥 Set your background image file name here
set_background("green-tea-bud-leaves-green-tea-plantations-morning.jpg")

# ==============================
# 🌱 Hero Section
# ==============================
st.markdown("""
<div style='text-align:center; padding:40px;'>
    <h1>🌱 Smart Farming AI Dashboard</h1>
    <p style='font-size:20px;'>
    Intelligent Agriculture System Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ==============================
# 🚀 Module Selection
# ==============================
st.subheader("🚀 Select Module")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, _ = st.columns(2)

with col1:
    if st.button("🌾 Crop Recommendation", use_container_width=True):
        st.switch_page("pages/Crop_Recommendation.py")

with col2:
    if st.button("📈 Yield Prediction", use_container_width=True):
        st.switch_page("pages/Yield_Prediction.py")

with col3:
    if st.button("💰 Price Prediction", use_container_width=True):
        st.switch_page("pages/Price_Prediction.py")

with col4:
    if st.button("🧪 Fertilizer Recommendation", use_container_width=True):
        st.switch_page("pages/Fertilizer_Recommendation.py")

with col5:
    if st.button("📜 Insurance Prediction", use_container_width=True):
        st.switch_page("pages/Insurance_Prediction.py")

st.divider()

st.success("You can also use the sidebar for navigation.")
