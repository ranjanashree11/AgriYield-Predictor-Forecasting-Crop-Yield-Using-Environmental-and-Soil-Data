# Agriyield--Predictor
ML-based crop yield prediction system
# 🌾 AgriYield Predictor

AI-based crop yield prediction system using ML.

## 🔹 Features

- Predict crop yield using machine learning models
- Input parameters include:
  - Soil information: N, P, K values
  - Weather conditions: Temperature, rainfall, humidity, wind, solar radiation
  - Location data: Region, crop, soil type
- Don’t Know NPK / Auto-fill Mode:
  - Farmers can still get yield predictions even if they don’t know NPK or total nutrient values
  - Auto-fills default or estimated values and marks them in reports and history
- Visual analytics:
  - Graphs for environmental factors and yield trends
- History tracking: Store previous predictions and download as CSV
- Report download: Generate and download yield prediction reports
- Actionable recommendations based on predicted yield
- 
- ## 🛠️ Technologies Used
- Python – Programming language
- Streamlit – Web dashboard
- Scikit-learn – Machine learning
- Pandas, NumPy – Data handling
- Matplotlib, Seaborn – Visualizations
- HTML/CSS – Dashboard styling

## 🚀 Run Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AgriYieldProject.git
cd AgriYieldProject

## Run Locally
```bash
streamlit run app.py
