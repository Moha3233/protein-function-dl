import pandas as pd
from data_preprocessing import load_and_clean, split_data
from feature_engineering import extract_features, scale_features
from train import train_model
from evaluate import evaluate

# Load data
df = load_and_clean("data/proteins.csv")

X = extract_features(df['sequence'])
y = df['label'].values

# Train / Val / Test split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Feature scaling
X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

# Train model
model = train_model(X_train, y_train, X_val, y_val)

# Evaluate
acc, prec, rec, f1, auc = evaluate(model, X_test, y_test)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC-AUC: {auc:.3f}")
