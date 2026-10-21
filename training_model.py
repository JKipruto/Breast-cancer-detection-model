import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

model_file = "Breast_cancer_dataset2.csv"
try:
    if os.path.exists(model_file):
        print("File available")
        dataset = pd.read_csv(model_file)
        columns = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                   "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
                   ]
        dataset = dataset[columns]
        row, col = dataset.shape
        print(f"{row} rows, {col} columns")
        print("\n")
        print("===Data information and description===")
        print(dataset.info())
        print("\n")
        print(dataset.describe())
        print("\n")
        print(dataset.isnull().sum())
        print("\n")
        print(dataset.isna().sum())
        print("\n")
        print(f"Duplicated:{dataset.duplicated().sum()}")

        encoder = LabelEncoder()
        y = dataset["diagnosis"]
        y = encoder.fit_transform(y)

        x = dataset.drop(columns="diagnosis")

        plt.figure(figsize=(8, 8))
        plt.title("A count plot for Diagnosis")
        ax = sns.countplot(x=y)
        ax.bar_label(ax.containers[0])
        plt.xticks([0, 1], encoder.classes_)
        plt.xlabel("Diagnosis")
        plt.ylabel("Counts")
        plt.show()

        for col in x.columns:
            plt.figure(figsize=(6, 6))
            plt.title(f"BoxPlot for {col}")
            sns.boxplot(x=col, data=x)
            plt.title(f"{col}'s BoxPlot")
            plt.xlabel(col)
            plt.show()

    else:
        print("Upload the file since it isn't available")
except Exception as e:
    print(e)
