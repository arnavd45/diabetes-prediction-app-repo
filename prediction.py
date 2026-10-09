import os 
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

# load .env content to env files
load_dotenv()


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT/ os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# loading the trained model
logging.info("Loading trained model")
model = load(MODEL_PATH)



def predict(input_data:dict):
    df = pd.DataFrame([input_data])
    predictions = model.predict(df)[0]
    logging.info("Model provided a prediction")
    return predictions



## example 
#input_data = {
 #   "Pregnancies":2,
  #  "Glucose":120,
   # "BloodPressure":150,
    #"SkinThickness":25,
    #"Insulin":50,
    #"BMI":28.5,
    #"DiabetesPedigreeFunction":0.583,
    #"Age":30
#}    

#model_predict = predict(input_data=input_data)
#print(model_predict)
