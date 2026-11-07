
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Dataset location
DATASET_PATH = "dataset/gestures.csv"

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "Dataset not found! Run collect_data.py first."
    )

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Check dataset
print("Dataset loaded successfully!")
print("Total samples:", len(df))

print("\nGesture distribution:")
print(df["label"].value_counts())

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Evaluation")
print("-------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": list(X.columns)
    },
    "models/gesture_model.pkl"
)

print("\nModel successfully saved!")
print("Location: models/gesture_model.pkl")
