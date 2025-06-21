# model_utils.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# ----------- Preprocessing Function -----------

def get_bmi_category(height_cm, weight_kg):
    if height_cm == 0 or weight_kg == 0:
        return "Normal"
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Overweight"

def preprocess_and_train_model():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df_processed = df.copy()
    df_processed = df_processed.drop("Person ID", axis=1)
    
    # Split BP
    bp_split = df_processed["Blood Pressure"].str.split("/", expand=True)
    df_processed["Systolic_BP"] = pd.to_numeric(bp_split[0])
    df_processed["Diastolic_BP"] = pd.to_numeric(bp_split[1])
    df_processed = df_processed.drop("Blood Pressure", axis=1)
    
    categorical_cols = ["Gender", "Occupation", "BMI Category"]
    df_processed["BMI Category"] = [
        get_bmi_category(h, w) for h, w in zip(df["Height (cm)"], df["Weight (kg)"])
    ]
    
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    X = df_processed.drop("Sleep Disorder", axis=1)
    y = df_processed["Sleep Disorder"].fillna("None")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Save model & training columns
    joblib.dump(model, "sleep_model.pkl")
    joblib.dump(X.columns.tolist(), "training_columns.pkl")

def load_model_and_columns():
    model = joblib.load("sleep_model.pkl")
    columns = joblib.load("training_columns.pkl")
    return model, columns

def predict_sleep_disorder(user_input):
    model, training_columns = load_model_and_columns()
    user_df = pd.DataFrame([user_input])
    
    user_df["BMI Category"] = get_bmi_category(
        user_input["Height (cm)"], user_input["Weight (kg)"]
    )
    
    bp_split = user_df["Blood Pressure"].str.split("/", expand=True)
    user_df["Systolic_BP"] = pd.to_numeric(bp_split[0])
    user_df["Diastolic_BP"] = pd.to_numeric(bp_split[1])
    
    user_df = user_df.drop(["Blood Pressure", "Height (cm)", "Weight (kg)"], axis=1)
    user_df = pd.get_dummies(user_df, columns=["Gender", "Occupation", "BMI Category"])
    
    user_df = user_df.reindex(columns=training_columns, fill_value=0)
    
    prediction = model.predict(user_df)[0]
    probabilities = model.predict_proba(user_df)[0]
    
    confidence_map = {cls: f"{prob:.2%}" for cls, prob in zip(model.classes_, probabilities)}
    
    return prediction, confidence_map
