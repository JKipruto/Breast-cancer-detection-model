import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

paths = ["feature_names.pkl", "breast_cancer_logreg.pkl"]

if all(os.path.exists(file) for file in paths):
    print("Files available")
    features = joblib.load("feature_names.pkl")
    logreg_pipe = joblib.load("breast_cancer_logreg.pkl")

    class Inputs(BaseModel):
        data: dict[str, float]

    app = FastAPI()

    @app.post("/predict")
    def predict(inputs: Inputs):
        missing = [f for f in features if f not in inputs.data]
        if missing:
            raise HTTPException(
                status_code=422, detail=f"Missing features: {missing}")

        df = pd.DataFrame([inputs.data])[features]
        pred = int(logreg_pipe.predict(df)[0])
        prob = float(logreg_pipe.predict_proba(df)[0, 1])

        return {
            "prediction": "malignant" if pred == 1 else "benign",
            "probability_malignant": round(prob, 4),
        }

else:
    raise FileNotFoundError("Files Unavailable")
