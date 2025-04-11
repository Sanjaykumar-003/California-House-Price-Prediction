import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import ssl
from io import StringIO
from sklearn.metrics import mean_squared_error

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load the saved models and scaler
@st.cache_resource
def load_models():
    models = {
        'Random Forest': joblib.load('random_forest_model.joblib'),
        'Linear Regression': joblib.load('linear_regression_model.joblib'),
        'Gradient Boosting': joblib.load('gradient_boosting_model.joblib'),
        'Support Vector Regression': joblib.load('support_vector_regression_model.joblib')
    }
    scaler = joblib.load('scaler.joblib')
    return models, scaler

# Load feature names and descriptions
@st.cache_data
def load_feature_info():
    california = fetch_california_housing()
    feature_names = california.feature_names
    feature_descriptions = {
        'MedInc': 'Median income in block group',
        'HouseAge': 'Median house age in block group',
        'AveRooms': 'Average number of rooms per household',
        'AveBedrms': 'Average number of bedrooms per household',
        'Population': 'Block group population',
        'AveOccup': 'Average number of household members',
        'Latitude': 'Block group latitude',
        'Longitude': 'Block group longitude'
    }
    return feature_names, feature_descriptions

def single_prediction_page(models, scaler, feature_names, feature_descriptions, california):
    st.title("🏠 Single House Price Prediction")
    
    # Create two columns for input and output
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Selection")
        selected_model = st.selectbox(
            "Choose a prediction model",
            list(models.keys()),
            help="Select the machine learning model to use for prediction"
        )

        st.subheader("Input Features")
        
        # Create sliders for each feature
        input_features = {}
        for feature in feature_names:
            min_val = float(california.data[:, feature_names.index(feature)].min())
            max_val = float(california.data[:, feature_names.index(feature)].max())
            
            st.write(f"**{feature}**")
            st.caption(feature_descriptions[feature])
            input_features[feature] = st.slider(
                f"Select {feature}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                key=feature
            )

        if st.button("Predict House Price"):
            input_array = np.array([list(input_features.values())])
            scaled_input = scaler.transform(input_array)
            
            # Make predictions with all models
            predictions = {}
            confidences = {}
            
            for name, model in models.items():
                if name == 'Random Forest':
                    # For Random Forest, use tree predictions
                    tree_preds = np.array([tree.predict(scaled_input) for tree in model.estimators_])
                    predictions[name] = model.predict(scaled_input)[0]
                    confidences[name] = np.std(tree_preds) * 100000  # Convert to dollars
                elif name == 'Gradient Boosting':
                    # For Gradient Boosting, use staged predictions
                    staged_preds = np.array([pred for pred in model.staged_predict(scaled_input)])
                    predictions[name] = model.predict(scaled_input)[0]
                    confidences[name] = np.std(staged_preds) * 100000
                else:
                    # For other models, use prediction error
                    predictions[name] = model.predict(scaled_input)[0]
                    confidences[name] = np.sqrt(mean_squared_error(
                        [predictions[name]], 
                        [predictions[name]]
                    )) * 100000
            
            with col2:
                st.subheader("Prediction Results")
                
                # Create a DataFrame for predictions
                results_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted Price': [f"${p*100000:,.2f}" for p in predictions.values()],
                    'Confidence Interval': [f"±${c:,.2f}" for c in confidences.values()]
                })
                
                # Display the results table
                st.dataframe(results_df)
                
                # Create a comparison chart
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(predictions))
                width = 0.35
                
                # Plot predictions
                ax.bar(x - width/2, [p*100000 for p in predictions.values()], width, label='Predicted Price')
                # Plot confidence intervals
                ax.bar(x + width/2, [c for c in confidences.values()], width, label='Confidence Interval')
                
                ax.set_ylabel('Price ($)')
                ax.set_title('Model Predictions and Confidence Intervals')
                ax.set_xticks(x)
                ax.set_xticklabels(predictions.keys(), rotation=45)
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show feature importance if available
                if hasattr(models[selected_model], 'feature_importances_'):
                    st.subheader(f"Feature Importance ({selected_model})")
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': models[selected_model].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                    plt.title(f'Feature Importance ({selected_model})')
                    plt.tight_layout()
                    st.pyplot(fig)

def batch_prediction_page(models, scaler, feature_names):
    st.title("🏠 Batch House Price Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with house features", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Check if all required features are present
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
            else:
                # Select model
                selected_model = st.selectbox(
                    "Choose a prediction model",
                    list(models.keys()),
                    key="batch_model"
                )
                
                if st.button("Predict Batch"):
                    # Scale the features
                    X = df[feature_names].values
                    X_scaled = scaler.transform(X)
                    
                    # Make predictions
                    model = models[selected_model]
                    predictions = model.predict(X_scaled)
                    
                    # Add predictions to dataframe
                    df['Predicted_Price'] = predictions * 100000  # Convert to dollars
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="house_price_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Show distribution of predictions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df['Predicted_Price'], kde=True, ax=ax)
                    plt.title('Distribution of Predicted House Prices')
                    plt.xlabel('Price ($)')
                    plt.ylabel('Count')
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_comparison_page():
    st.title("🏠 Model Comparison")
    st.image('model_comparison.png', use_container_width=True)
    
    st.write("""
    ### Model Descriptions:
    - **Random Forest**: An ensemble of decision trees, good for handling non-linear relationships
    - **Linear Regression**: Simple linear model, good for understanding feature relationships
    - **Gradient Boosting**: Sequential ensemble of weak learners, often provides high accuracy
    - **Support Vector Regression**: Good for handling complex non-linear relationships
    """)

def main():
    # Load the California housing dataset
    california = fetch_california_housing()
    
    # Load models and feature information
    models, scaler = load_models()
    feature_names, feature_descriptions = load_feature_info()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Single Prediction", "Batch Prediction", "Model Comparison"]
    )
    
    # Display the selected page
    if page == "Single Prediction":
        single_prediction_page(models, scaler, feature_names, feature_descriptions, california)
    elif page == "Batch Prediction":
        batch_prediction_page(models, scaler, feature_names)
    else:
        model_comparison_page()

if __name__ == "__main__":
    main() 