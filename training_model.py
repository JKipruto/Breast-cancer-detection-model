import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

model_file = "Breast_cancer_dataset2.csv"
try:
    if os.path.exists(model_file):
        random_state = 42
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
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
            x, y, test_size=0.2, random_state=random_state, stratify=y)

        try:
            scorings = [f_classif, mutual_info_classif]
            for scoring in scorings:
                selector = SelectKBest(
                    score_func=scoring,
                    k=16
                )
                selector.fit_transform(x_train_1, y_train_1)
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
        print(data_corr.columns)

        plt.figure(figsize=(12, 12))
        plt.title("Correlation Heatmap")
        sns.heatmap(data_corr.corr(), annot=True, cmap="coolwarm",
                    fmt=".2f", vmin=-1, vmax=1)
        plt.show()

        x = data_corr.drop("diagnosis", axis=1)
        y = data_corr["diagnosis"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_state, stratify=y)

        scaler = StandardScaler()

        def Build_Pipeline(model):
            return Pipeline([("scaler", scaler), ("model", model)])

        voting_estimators = [
            ("logReg", Build_Pipeline(LogisticRegression(
                max_iter=1000, random_state=random_state))),
            ("svc", Build_Pipeline(SVC(probability=True, random_state=random_state))),
            ("rf", Build_Pipeline(RandomForestClassifier(random_state=random_state)))
        ]

        max_depth = np.arange(2, 20, 1)
        leaf_nodes = np.arange(2, 20, 1)
        kernel = ["linear", "poly", "rbf"]
        gamma = ["scale", "auto"]
        model_c = np.logspace(-2, 2, 10)
        no_neighbors = np.arange(2, 20)
        weights = ["distance", "uniform"]
        p = [1, 2]
        n_estimators = np.arange(50, 310, 10)
        learning_rate = np.logspace(-2, 0, 8)
        voting_type = ["hard", "soft"]

        search_config = {  # "model_name":(model,kind of cv,parameters in dict,iteration-->for randomizedCV)
            "logreg": (LogisticRegression(max_iter=1000, random_state=random_state), "grid", {"model__C": model_c}, None),
            "Destree": (DecisionTreeClassifier(random_state=random_state), "grid", {"model__max_depth": max_depth, "model__max_leaf_nodes": leaf_nodes}, None),
            "SVC": (SVC(random_state=random_state), "grid", {"model__kernel": kernel, "model__gamma": gamma, "model__C": model_c}, None),
            "knn": (KNeighborsClassifier(), "grid", {"model__n_neighbors": no_neighbors, "model__weights": weights, "model__p": p}, None),
            "rf": (RandomForestClassifier(random_state=random_state), "random", {"model__max_depth": max_depth, "model__max_leaf_nodes": leaf_nodes, "model__n_estimators": n_estimators}, 20),
            "bag": (BaggingClassifier(random_state=random_state, estimator=DecisionTreeClassifier(random_state=random_state)), "random", {"model__n_estimators": n_estimators}, 20),
            "adaboost": (AdaBoostClassifier(random_state=random_state, estimator=DecisionTreeClassifier(random_state=random_state, max_depth=2)), "random", {"model__n_estimators": n_estimators, "model__learning_rate": learning_rate}, 20),
            "voting": (VotingClassifier(estimators=voting_estimators), "grid", {"model__voting": voting_type}, None)
        }

        best_estimator = {}
        tuned_value = {}
        best_score = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=random_state)
        for name, (model, kind, param, iteration) in search_config.items():
            pipe = Build_Pipeline(model)
            if kind == "grid":
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=param,
                    scoring="recall",
                    cv=cv,
                    n_jobs=2,
                    verbose=2
                )

            else:
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param,
                    n_iter=iteration,
                    scoring="recall",
                    cv=cv,
                    n_jobs=2,
                    verbose=3,
                    random_state=random_state
                )

            search.fit(x_train, y_train)
            best_estimator[name] = search.best_estimator_
            tuned_value[name] = search.best_params_
            best_score[name] = search.best_score_
        metrics = []
        cm = {}
        for name, model in best_estimator.items():
            pred = model.predict(x_test)
            metrics.append({
                "name": name,
                "precision score": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "Acuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred)
            })
            cr = classification_report(y_test, pred)
            print(cr)
            cm[name] = confusion_matrix(y_test, pred)

        metrics_df = pd.DataFrame(metrics)
        print(metrics_df)

        for name, values in cm.items():
            print(name)
            print(values)

        for name, report in cm.items():
            plt.figure(figsize=(6, 6))
            plt.title(f"Confusion Matrix of {name}")
            sns.heatmap(report, annot=True, cmap="coolwarm", fmt="d",
                        xticklabels=encoder.classes_, yticklabels=encoder.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
    else:
        print("Upload the file since it isn't available")
except FileNotFoundError as e:
    print(e)
