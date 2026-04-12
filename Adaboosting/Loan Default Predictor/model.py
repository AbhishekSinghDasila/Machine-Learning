import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Pre-Split Data
# -----------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# -----------------------------
# Store Loan_ID separately
# -----------------------------
test_ids = test_df["Loan_ID"]

# Drop ID column
train_df.drop("Loan_ID", axis=1, inplace=True)
test_df.drop("Loan_ID", axis=1, inplace=True)

# -----------------------------
# Handle Missing Values
# -----------------------------
train_df.fillna(train_df.mode().iloc[0], inplace=True)
test_df.fillna(train_df.mode().iloc[0], inplace=True)

# -----------------------------
# Identify categorical columns
# -----------------------------
categorical_cols = train_df.select_dtypes(include=['object']).columns

# Remove target column if present
categorical_cols = [col for col in categorical_cols if col != "Loan_Status"]

# -----------------------------
# Apply Label Encoding properly
# -----------------------------
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()

    # Fit on train
    train_df[col] = le.fit_transform(train_df[col])

    # Apply same mapping on test
    test_df[col] = le.transform(test_df[col])

    le_dict[col] = le
# -----------------------------
# Split Features
# -----------------------------
X_train = train_df.drop("Loan_Status", axis=1)
y_train = train_df["Loan_Status"]

X_test = test_df.copy()

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train AdaBoost Model
# -----------------------------
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
feature_names = train_df.drop("Loan_Status", axis=1).columns

plt.figure()
plt.barh(feature_names, importances)
plt.title("Feature Importance (AdaBoost)")
plt.show()

# -----------------------------
# Save Model + Preprocessing
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_dict, open("encoders.pkl", "wb"))

# -----------------------------
# Predict on Test Data
# -----------------------------
test_preds = model.predict(X_test)

# Save predictions (important for Kaggle)
output = pd.DataFrame({
    "Loan_ID": test_ids,
    "Loan_Status": test_preds
})

output.to_csv("submission.csv", index=False)

print("✅ Model trained and submission file created!")