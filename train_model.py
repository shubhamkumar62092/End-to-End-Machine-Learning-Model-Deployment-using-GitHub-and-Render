"""
train_model.py
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

Dataset: heart.csv (Kaggle - johnsmith88/heart-disease-dataset)
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------
# TASK 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and the target variable
target_column = "target"
numerical_features = [col for col in df.columns if col != target_column]

print("\nNumerical features:")
print(numerical_features)
print("\nTarget variable:", target_column)

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ---------------------------------------------------------------------
# TASK 2: Model Development
# ---------------------------------------------------------------------

# Feature scaling (helps most classifiers, kept in the saved pipeline)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model: Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model AND the scaler + feature order together,
# so app.py can reproduce the exact preprocessing at inference time.
artifact = {
    "model": model,
    "scaler": scaler,
    "feature_names": numerical_features,
    "accuracy": accuracy,
}
joblib.dump(artifact, "model.pkl")
print("\nSaved trained model to model.pkl")
