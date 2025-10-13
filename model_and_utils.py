# model_and_utils.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import shap
import joblib

FEATURES = ["vegetation", "soil_moisture", "distance_to_road", "conflict_intensity", "elevation"]

def train_rf(df, test_size=0.2, random_state=42):
    X = df[FEATURES]
    y = df["mine"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, model.predict(X_test))
    metrics = {"auc": auc, "accuracy": acc}
    return model, metrics

def predict_grid(model, df):
    """
    Given model and dataframe with FEATURES + lon,lat, return predicted probability column.
    """
    X = df[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["risk_proba"] = proba
    return df_out

def incremental_update(model, df_existing, df_new):
    """
    Simple incremental update: combine datasets and retrain model.
    For small-scale demo this is fine; for large data use partial_fit classifiers.
    """
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    model_new, metrics = train_rf(df_combined, test_size=0.2)
    return model_new, metrics

def explain_model_shap(model, df_sample, nsamples=100):
    """
    Compute SHAP values. Returns shap.Explainer and expected values.
    """
    X = df_sample[FEATURES]
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)  # can be heavy; use subset if needed
    return explainer, shap_values

def save_model(model, path="rf_model.joblib"):
    joblib.dump(model, path)

def load_model(path="rf_model.joblib"):
    return joblib.load(path)
