import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

def load_and_preprocess_data():
    # Load the California Housing dataset
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = pd.Series(california.target, name='MedHouseVal')
    
    # Create a DataFrame with features and target
    df = pd.concat([X, y], axis=1)
    
    # Display basic information about the dataset
    print("\nDataset Information:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return X, y

def train_models(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Calculate confidence intervals
        if name == 'Random Forest':
            # For Random Forest, use tree predictions
            tree_predictions = np.array([tree.predict(X_test_scaled) for tree in model.estimators_])
            confidence = np.std(tree_predictions, axis=0)
        elif name == 'Gradient Boosting':
            # For Gradient Boosting, use stage predictions
            stage_predictions = np.array([m.predict(X_test_scaled) for m in model.estimators_])
            confidence = np.std(stage_predictions, axis=0)
        else:
            # For other models, use prediction error
            confidence = np.sqrt(mse) * np.ones_like(y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confidence': confidence.mean()
        }
        
        print(f"{name} Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        print(f"Cross-validation R2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"Average Confidence Interval: ±{confidence.mean():.4f}")
    
    return results, scaler

def plot_model_comparison(results):
    # Create comparison plot
    metrics = ['rmse', 'r2', 'cv_mean', 'confidence']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in results]
        if metric == 'rmse':
            # Lower RMSE is better
            axes[i].bar(results.keys(), values, color='red')
            axes[i].set_title('RMSE (Lower is Better)')
        elif metric == 'confidence':
            # Lower confidence interval is better
            axes[i].bar(results.keys(), values, color='purple')
            axes[i].set_title('Average Confidence Interval (Lower is Better)')
        else:
            # Higher R2 is better
            axes[i].bar(results.keys(), values, color='green')
            axes[i].set_title('R2 Score (Higher is Better)')
        
        axes[i].set_xticklabels(results.keys(), rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def save_models(results, scaler):
    # Save all models and scaler
    for name, result in results.items():
        joblib.dump(result['model'], f'{name.lower().replace(" ", "_")}_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("\nModels and scaler saved successfully!")

def main():
    print("Starting House Price Prediction Pipeline...")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Train and evaluate models
    results, scaler = train_models(X, y)
    
    # Plot model comparison
    plot_model_comparison(results)
    
    # Save models
    save_models(results, scaler)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 