from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load(r"C:\Users\mhdto\OneDrive\Documents\Project DWT\backend\gb_model.joblib")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the wait-time predictor API!"}

class InputData(BaseModel):
    arrival_delta_min: float
    est_service_time_min: float
    lambda_per_min: float
    mu_per_min: float
    servers: int
    Wq_analytic_min: float
    Wq_simulated_480min: float
    NumScannersUsedToday: int
    InProgressSize: float
    SumInProgress: float
    NumCompletedToday: int
    SumHowEarlyWaiting: float
    AvgHowEarlyWaiting: float
    LineCount0: int
    LineCount1: int
    LineCount2: int
    LineCount3: int
    LineCount4: int
    SumWaits: float
    DelayCount: int
    DelayCountLastHour: int
    AvgWaitByTaskTypeLine: float
    DelayedInLine: int
    is_holiday: bool
    is_event: bool
    dow_sin: float
    dow_cos: float
    hour_sin: float
    hour_cos: float
    recent_wait_30min: float
    recent_wait_60min: float

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([list(data.model_dump().values())])
        prediction = model.predict(input_array)[0]
        return {"predicted_wait_time_minutes": round(float(prediction), 2)}
    except Exception as e:
        return {"error": str(e)}