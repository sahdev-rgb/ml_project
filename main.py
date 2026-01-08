from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'salary_predict.pkI')

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print(f" Error loading model: {e}")

class SalaryInput(BaseModel):
    age: float
    gender: str
    education: str
    job_title: str
    experience: float

@app.post("/predict")
def predict(data: SalaryInput):
    try:
      
        features = np.array([[data.age, data.gender, data.education, data.job_title, data.experience]], dtype=object)
        
        prediction = model.predict(features)
        return {"salary": round(float(prediction[0]), 2)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

