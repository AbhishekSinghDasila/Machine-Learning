import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset (use correct path if needed)
df = pd.read_csv("Churn_dataset.csv")

# Drop ID
df.drop("customerID", axis=1, inplace=True)

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Fix TotalCharges (IMPORTANT)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values (ONLY numeric)
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save columns
pickle.dump(X.columns, open("columns.pkl", "wb"))

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved successfully!")