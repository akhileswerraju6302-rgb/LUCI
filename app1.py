# ============================================
# INSURANCE FRAUD DETECTION PROJECT
# ============================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ============================================
# 2. LOAD DATASET
# ============================================
df = pd.read_csv("insurance_fraud.csv")  # change file name if needed

print("First 5 rows:")
print(df.head())


# ============================================
# 3. DATA UNDERSTANDING
# ============================================
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================
# 4. DATA CLEANING
# ============================================

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)


# ============================================
# 5. ENCODE CATEGORICAL DATA
# ============================================
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


# ============================================
# 6. DEFINE FEATURES AND TARGET
# ============================================
target = 'fraud_reported'   # change if different

X = df.drop(columns=[target])
y = df[target]


# ============================================
# 7. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# 8. TRAIN MODELS
# ============================================

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)


# ============================================
# 9. MODEL EVALUATION FUNCTION
# ============================================
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Results")
    print("--------------------------------")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# Evaluate all models
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("XGBoost", y_test, xgb_pred)


# ============================================
# 10. FRAUD RISK SCORING
# ============================================

fraud_prob = rf_model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results['Actual'] = y_test
results['Fraud_Probability'] = fraud_prob

# Show high risk claims
high_risk = results[results['Fraud_Probability'] > 0.7]

print("\nHigh Risk Claims:")
print(high_risk.head())


# ============================================
# 11. FEATURE IMPORTANCE
# ============================================

importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features:")
print(importance_df.head())


# ============================================
# 12. VISUALIZATION
# ============================================

sns.countplot(x=y)
plt.title("Fraud vs Non-Fraud")
plt.show()


# ============================================
# 13. PREDICTION FUNCTION
# ============================================

def predict_claim(model, input_data):
    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "Fraud Prediction": int(prediction),
        "Fraud Probability": float(probability)
    }


# ============================================
# 14. EXAMPLE PREDICTION
# ============================================

# Example input (change values based on your dataset)
sample_input = X.iloc[0].to_dict()

result = predict_claim(rf_model, sample_input)

print("\nSample Prediction:")
print(result)

# ============================================
# END OF PROJECT
# ============================================
