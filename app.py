# Licensed under the MIT License. See LICENSE file for more info.

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.markdown("""
    <style>
    .stApp {
        background-image:  linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)) ,url("https://media.istockphoto.com/id/1302922398/photo/top-view-of-bright-pink-ribbon-on-dark-wood-background-breast-cancer-awareness-and-womens.jpg?s=612x612&w=0&k=20&c=wpD3n37ipEmjV72utgh6SflUPvyzjDOu203_tlIbCMU=");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    h1 {
        color: #FFD700;  /* Gold */
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .glow-text {
        font-size: 50px;
        color: #ffffff;
        text-align: center;
        text-shadow: 0 0 10px #00cfff, 0 0 20px #00cfff, 0 0 30px #00cfff;
        font-weight: bold;
    }
    </style>
    <div class="glow-text">🎗 Breast Cancer Risk Predictor</div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model_and_scaler():
    model = load_model("breast_cancer_ann_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# Page configuration
st.title("Breast Cancer Prediction using ANN 🤖🧠")
st.write("Provide the required inputs below to predict whether the tumor is **Benign (0)** or **Malignant (1)**.")

st.markdown("""
    <style>
    /* Sidebar custom style */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 50, 0.8);  /* Dark blue-ish tone */
        color: white;
    }

    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white;
    }

    /* Optional: make sidebar title/headings colored */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00cfff;  /* Light cyan */
    }

    /* Optional: control scrollbar style inside sidebar */
    ::-webkit-scrollbar-thumb {
        background: #00cfff;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar.expander("📁 Project Intro"):
    st.markdown("- **This is a Breast Cancer Risk Prediction web app using an Artificial Neural Network (ANN)." \
    "It takes medical input features and predicts the likelihood of a Breast Cancer.**")
 

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **IBM Certifed Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of Large Language Models (LLMs)**")
    st.markdown("- **Have expertise in EDA, ML, Reinforcement Learning, ANN, CNN, CV, RNN, NLP, LLMs.**")
    st.markdown("[💼Visit Rayyan's LinkedIn Profile](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("- **Numpy**")
    st.markdown("- **Pandas**")
    st.markdown("- **Matplotlib**")
    st.markdown("- **Seaborn**")
    st.markdown("- **Scikit Learn**")
    st.markdown("- **TensorFlow, Keras, Pickle**")
    st.markdown("- **Streamlit**")

# Input features
feature_names = [
    'radius_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'texture_se', 'perimeter_se', 'radius_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst'
]

user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0.0, format="%.3f")
    user_input.append(val)

# Predict button
if st.button("Predict"):
    input_array = np.array([user_input])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    output = (prediction > 0.5).astype(int)

    st.subheader("🔎 Prediction Result:")
    if output[0][0] == 1:
        st.error("The tumor is **Malignant (Cancerous)** ❌")
        st.snow()
    else:
        st.success("The tumor is **Benign (Non-cancerous)** ✅")
        st.balloons()
