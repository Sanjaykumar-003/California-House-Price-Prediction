import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf')
}

# Train and save models
r2_scores = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2_scores[name] = r2_score(y_test, y_pred)
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Create model comparison plot
plt.figure(figsize=(10, 6))
plt.bar(r2_scores.keys(), r2_scores.values())
plt.title('Model Comparison (R² Score)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nModel R² Scores:")
for name, score in r2_scores.items():
    print(f"{name}: {score:.4f}")
print("\nAll models and files have been saved successfully!") 