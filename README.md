# 🎗 Breast Cancer Risk Predictor using ANN

This project is a **web application built with Streamlit** that predicts the risk of **Breast Cancer** (Benign or Malignant) using an **Artificial Neural Network (ANN)** trained on diagnostic medical features.

![Banner](https://media.istockphoto.com/id/1302922398/photo/top-view-of-bright-pink-ribbon-on-dark-wood-background-breast-cancer-awareness-and-womens.jpg?s=612x612&w=0&k=20&c=wpD3n37ipEmjV72utgh6SflUPvyzjDOu203_tlIbCMU=)

---

## 🚀 Features

- 🎯 Predicts **Benign (0)** or **Malignant (1)** tumors using 12 important medical features.
- 🌐 Interactive user interface with stylish design using Streamlit.
- 🧠 Powered by a trained ANN using TensorFlow & Keras.
- 📊 Trained on publicly available breast cancer dataset.
- ✅ Includes **standard scaler**, ANN `.keras` model, and input feature support.
- 📎 MIT License included.

---

## 🧪 Input Features Used

- `radius_mean`
- `compactness_mean`
- `concavity_mean`
- `concave points_mean`
- `texture_se`
- `perimeter_se`
- `radius_worst`
- `smoothness_worst`
- `compactness_worst`
- `concavity_worst`
- `concave points_worst`
- `symmetry_worst`

---

## 📂 Project Structure

├── app.py # Streamlit frontend and predictor
├── breast_cancer_ann_model.keras # Trained ANN model file
├── scaler.pkl # StandardScaler object (fitted)
├── requirements.txt # Dependencies list
├── LICENSE # MIT License
|── CSV Dataset File
└── README.md # Project documentation


---

## 🔧 Installation

### ✅ Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/breast-cancer-ann-app.git
cd breast-cancer-ann-app
```

### ✅ Step 2: Create virtual environment

```
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### ✅ Step 3: Install dependencies
```
pip install -r requirements.txt
```

### Run Streamlit App

```
streamlit run app.py
```

📦 Requirements
You can install dependencies using the provided requirements.txt file. Major packages include:

- streamlit
- tensorflow
- numpy
- scikit-learn
- pandas

📝 License
This project is licensed under the [MIT License]()
