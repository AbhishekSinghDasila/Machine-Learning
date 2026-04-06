# 📊 Customer Churn Prediction using Random Forest

## 🚀 Overview

This project predicts whether a customer is likely to churn (leave the service) using a Random Forest model.
It includes an interactive web application built with Streamlit for real-time predictions and visualization.

---

## 🎯 Features

* Predict customer churn in real-time
* Probability-based predictions (not just Yes/No)
* Feature importance visualization
* User-friendly dashboard
* Handles class imbalance using optimized model

---

## 🧠 Tech Stack

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

---

## 📂 Project Structure

```
churn-app/
│
├── app.py                # Streamlit app
├── train_model.py        # Model training script
├── model.pkl             # Trained model
├── columns.pkl           # Feature columns
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
```

---

## ⚙️ How It Works

1. Data preprocessing (handling missing values & encoding)
2. Train Random Forest model
3. Save model using pickle
4. Load model in Streamlit app
5. Take user input → Predict churn probability
6. Display results + charts

---

## 📊 Model Details

* Algorithm: Random Forest
* Handles imbalance using `class_weight="balanced"`
* Tuned parameters:

  * n_estimators = 200
  * max_depth = 12
  * min_samples_split = 5

---

## ▶️ Run Locally

### Step 1: Clone repo

```
git clone <your-repo-link>
cd churn-app
```

### Step 2: Install dependencies

```
pip install -r requirements.txt
```

### Step 3: Train model

```
python train_model.py
```

### Step 4: Run app

```
streamlit run app.py
```

---

## 🌐 Deployment

This app can be deployed on platforms like:

* Streamlit Cloud
* Render

---

## 📈 Sample Output

* Churn Probability Score
* Stay vs Churn Visualization
* Feature Importance Chart

---

## 🧾 Resume Highlight

Built and deployed a Machine Learning model using Random Forest to predict customer churn, with an interactive dashboard and real-time predictions.

---

## 📌 Future Improvements

* Add ROC Curve & AUC
* Use XGBoost for better performance
* Deploy with Docker
* Add database integration

---

## 👨‍💻 Author

Abhishek Singh Dasila
