import requests

# fastapi endpoint URL 
API_URL = "http://127.0.0.1:8000/predict"

payload = {
  "Pregnancies": 1,
  "Glucose": 120,
  "BloodPressure": 34,
  "SkinThickness": 122,
  "Insulin": 45,
  "BMI": 123,
  "DiabetesPedigreeFunction": 3.4,
  "Age": 34
}

# send  POST request
response = requests.post(API_URL, json=payload)

print("Status Code:",response.status_code)
print("Response Json:",response.json())
