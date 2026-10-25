import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
        print(f"Duplicated:{dataset.duplicated().sum()}")

        encoder = LabelEncoder()
        y = dataset["diagnosis"]
        y = encoder.fit_transform(y)
        print(encoder.classes_)

        x = dataset.drop(columns="diagnosis")
        dataset["diagnosis"] = y
        corr = dataset.corr()
        print(corr)
        f_scores = {}
        mi_scores = {}
        fatures = []

        try:
            scorings = [f_classif, mutual_info_classif]
            for scoring in scorings:
                selector = SelectKBest(
                    score_func=scoring,
                    k=16
                )
                x_new_features = selector.fit_transform(x, y)
                if scoring is f_classif:
                    f_scores["feature"] = x.columns
                    f_scores["score"] = selector.scores_
                    f_scores["selected"] = selector.get_support()
                    f_scores_df = pd.DataFrame(f_scores)
                    print(scoring)
                    print(f_scores_df.sort_values(by="score", ascending=False))
                    print("\n")

                else:
                    mi_scores["feature"] = x.columns
                    mi_scores["score"] = selector.scores_
                    mi_scores["selected"] = selector.get_support()
                    mi_scores_df = pd.DataFrame(mi_scores)
                    print(scoring)
                    print(mi_scores_df.sort_values(
                        by="score", ascending=False))
                    print("\n")

        except Exception as e:
            print(e)

        plt.figure(figsize=(8, 8))
        plt.title("A count plot for Diagnosis")
        ax = sns.countplot(x=y)
        ax.bar_label(ax.containers[0])
        plt.xticks([0, 1], encoder.classes_)
        plt.xlabel("Diagnosis")
        plt.ylabel("Counts")
        plt.show()

        features = mi_scores["feature"].to_list()
        print(features)
        x = x.drop(columns=["texture_worst", "concavity_se",
                   "texture_mean", "smoothness_worst", "symmetry_worst", "smoothness_mean", "compactness_se", "fractal_dimension_worst", "symmetry_mean", "fractal_dimension_se", "smoothness_se", "symmetry_se",
                            "fractal_dimension_mean", "texture_se"])
        for col in x.columns:
            plt.figure(figsize=(5, 4))
            plt.title(f"Skewness of {col}")
            sns.kdeplot(x=col, data=x)
            plt.xlabel(col)
            plt.show()
        columns_2 = ["diagnosis", "radius_mean", "perimeter_mean", "area_mean", "compactness_mean",
                     "concavity_mean", "concave points_mean", "radius_se", "perimeter_se",
                     "area_se", "concave points_se", "radius_worst", "perimeter_worst",
                     "area_worst", "compactness_worst", "concavity_worst",
                     "concave points_worst"]
        data_corr = dataset[columns_2]

        plt.figure(figsize=(12, 12))
        plt.title("Correlation Heatmap")
        sns.heatmap(data_corr.corr(), annot=True, cmap="coolwarm",
                    fmt=".2f", square=True, vmin=-1, vmax=1)
        plt.show()

    else:
        print("Upload the file since it isn't available")
except FileNotFoundError as e:
    print(e)
