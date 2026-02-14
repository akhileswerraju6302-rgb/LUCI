# ==============================
# ClaimWatch AI - Fraud Detection
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 2. Load Dataset
# Replace with your actual dataset path
df = pd.read_csv("insurance_claims.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# 3. Data Preprocessing

# Drop unnecessary columns if any
if 'policy_number' in df.columns:
    df.drop(columns=['policy_number'], inplace=True)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Define Features and Target
X = df.drop("fraud_reported", axis=1)   # Target column
y = df["fraud_reported"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 7. Model Training
# ==============================

# 1️⃣ Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# 2️⃣ Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 3️⃣ XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# ==============================
# 8. Model Evaluation Function
# ==============================

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# 9. Evaluate All Models
evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree")
evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")


# ==============================
# 10. Risk Scoring System
# ==============================

def generate_risk_score(model, X_data):
    probabilities = model.predict_proba(X_data)[:, 1]
    risk_category = []

    for prob in probabilities:
        if prob < 0.3:
            risk_category.append("Low Risk")
        elif prob < 0.7:
            risk_category.append("Medium Risk")
        else:
            risk_category.append("High Risk")

    return probabilities, risk_category


# Example: Risk scoring using best model (XGBoost)
risk_prob, risk_level = generate_risk_score(xgb_model, X_test_scaled)

risk_df = pd.DataFrame({
    "Fraud_Probability": risk_prob,
    "Risk_Level": risk_level
})

print("\nSample Risk Scoring Output:")
print(risk_df.head())
