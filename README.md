# 🏠 California House Price Prediction

A Streamlit web application for predicting house prices in California using various machine learning models.

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Navigation Guide](#-navigation-guide)
- [Models](#-models)
- [Dataset](#-dataset)
- [Contributing](#-contributing)

## ✨ Features
- Single house price prediction
- Batch prediction with CSV file upload
- Multiple machine learning models
- Model comparison and performance metrics
- Interactive feature importance visualization
- Confidence intervals for predictions

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd House-Price-Prediction-using-Streamlit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)

## 🧭 Navigation Guide

### Single Prediction Page
- Select a machine learning model from the dropdown
- Adjust feature values using sliders:
  - MedInc: Median income in block group
  - HouseAge: Median house age in block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude
- Click "Predict House Price" to get the prediction
- View prediction results and feature importance (if available)

### Batch Prediction Page
- Upload a CSV file containing house features
- Select a prediction model
- Click "Predict Batch" to process all houses
- View and download prediction results
- See distribution of predicted prices

### Model Comparison Page
- View performance comparison of all models
- Read model descriptions and use cases
- Analyze model strengths and weaknesses

## 🤖 Models

The application includes four machine learning models:

1. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Provides confidence intervals
   - Shows feature importance

2. **Linear Regression**
   - Simple linear model
   - Good for understanding feature relationships
   - Fast predictions

3. **Gradient Boosting**
   - Sequential ensemble of weak learners
   - Often provides high accuracy
   - Handles complex patterns

4. **Support Vector Regression**
   - Good for complex non-linear relationships
   - Robust to outliers
   - Works well with high-dimensional data

## 📊 Dataset

The application uses the California Housing dataset, which includes:
- 8 numerical features
- Target variable: Median house value
- Approximately 20,640 samples
- Features include income, age, rooms, population, etc.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.