from fastapi import FastAPI
from pydantic import BaseModel

from prediction import predict


app = FastAPI(title="ML prediction API")

# define input schema
class PredictionInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# ml prediction app
@app.post("/predict")
def predict_diabetes(input_data:PredictionInput):
    prediction = predict(input_data.model_dump())
    return {
        "Prediction": int(prediction)
    }


