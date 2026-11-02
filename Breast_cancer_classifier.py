import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os

paths = ["feature_names.pkl", "breast_cancer_logreg.pkl"]

if all(os.path.exists(file) for file in paths):
    print("Files available")
    features = joblib.load("feature_names.pkl")
    logreg_pipe = joblib.load("breast_cancer_logreg.pkl")
    print(features)
else:
    print("Files Unavailable")
