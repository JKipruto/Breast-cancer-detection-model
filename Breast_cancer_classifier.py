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
    print(features)

# 'radius_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'radius_se', 'radius_worst', 'perimeter_worst', 'area_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst'
    class Inputs(BaseModel):
        radius_mean: float
        compactness_mean: float
        concavity_mean: float
        concave_points_mean: float
        radius_se: float
        radius_worst: float
        perimeter_worst: float
        area_worst: float
        compactness_worst: float
        concavity_worst: float
        concave_points_worst: float

    app = FastAPI()

    @app.get("/")
    def health():
        return {"status": "ok", "n_features": len(features)}

    @app.post("/predict")
    def predict(inputs: Inputs):
        missing = [f for f in features if f not in inputs.data]
        if missing:
            raise HTTPException(
                status_code=422, detail=f"Missing features: {missing}")

        df = pd.DataFrame([inputs.model_dump()])
        pred = int(logreg_pipe.predict(df)[0])
        prob = float(logreg_pipe.predict_proba(df)[0, 1])

        return {
            "prediction": "malignant" if pred == 1 else "benign",
            "probability_malignant": round(prob, 4),
        }

else:
    raise FileNotFoundError("Files Unavailable")
