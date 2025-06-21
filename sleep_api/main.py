# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict_sleep_disorder, preprocess_and_train_model

app = FastAPI(title="Sleep Disorder Predictor API")

# Only run this once to train and save model
# preprocess_and_train_model()

# ---- Pydantic schema ----
class UserInput(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    Height_cm: int
    Weight_kg: int
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int

@app.get("/")
def root():
    return {"message": "Welcome to the Sleep Disorder Predictor API"}

@app.post("/predict")
def predict(user: UserInput):
    user_dict = {
        "Gender": user.Gender,
        "Age": user.Age,
        "Occupation": user.Occupation,
        "Sleep Duration": user.Sleep_Duration,
        "Quality of Sleep": user.Quality_of_Sleep,
        "Physical Activity Level": user.Physical_Activity_Level,
        "Stress Level": user.Stress_Level,
        "Height (cm)": user.Height_cm,
        "Weight (kg)": user.Weight_kg,
        "Blood Pressure": user.Blood_Pressure,
        "Heart Rate": user.Heart_Rate,
        "Daily Steps": user.Daily_Steps
    }
    
    prediction, confidence = predict_sleep_disorder(user_dict)
    return {
        "predicted_sleep_disorder": prediction,
        "confidence_scores": confidence
    }
