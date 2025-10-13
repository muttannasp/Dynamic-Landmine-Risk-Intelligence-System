# model_and_utils.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
import shap
import joblib

FEATURES = ["vegetation", "soil_moisture", "distance_to_road", "conflict_intensity", "elevation"]

def train_rf(df, test_size=0.2, random_state=42, cv=5):
    """
    Train a RandomForest with improved defaults and report cross-validated AUC + test metrics.
    Returns: model, metrics(dict), feature_importances (pd.Series)
    """
    X = df[FEATURES].copy()
    y = df["mine"].copy()

    # Train / test split (stratify to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Stronger RF baseline with class weighting to handle imbalance
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Test metrics
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else float("nan")
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Cross-validated AUC on training set (to estimate expected generalization)
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_auc_mean = float(np.nanmean(cv_scores))
        cv_auc_std = float(np.nanstd(cv_scores))
    except Exception:
        cv_auc_mean = float("nan")
        cv_auc_std = float("nan")

    feature_importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    metrics = {
        "test_auc": test_auc,
        "test_accuracy": test_acc,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return model, metrics, feature_importances

def predict_grid(model, df):
    """
    Add risk_proba column using the given model. Returns a copy of dataframe.
    """
    X = df[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["risk_proba"] = proba
    return df_out

def incremental_update(model, df_existing, df_new):
    """
    Simple incremental update: combine datasets and retrain.
    (Small-demo approach; for production use incremental learners or partial_fit)
    """
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    model_new, metrics, fi = train_rf(df_combined, test_size=0.2)
    return model_new, metrics, fi

def explain_model_shap(model, df_sample):
    """
    Produce SHAP values array for class=1 where possible.
    Returns tuple (explainer, shap_values_arr, X_sample_df)
    shap_values_arr will be shape (n_samples, n_features).
    """
    X = df_sample[FEATURES].copy()
    # Prefer TreeExplainer for tree models (fast)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        
        # Handle different return formats from TreeExplainer
        if isinstance(sv, list) and len(sv) > 1:
            # List format: [class0_values, class1_values]
            shap_values = sv[1]  # Use class 1 (positive class)
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 3 and sv.shape[-1] == 2:
                # 3D array: (n_samples, n_features, n_classes)
                shap_values = sv[:, :, 1]  # Extract class 1 values
            else:
                # 2D array: (n_samples, n_features)
                shap_values = sv
        else:
            shap_values = sv
            
    except Exception as e:
        print(f"TreeExplainer failed: {e}, trying fallback...")
        # Fallback to general Explainer
        try:
            explainer = shap.Explainer(model, X)
            ev = explainer(X)
            # ev may be an Explanation object; attempt to extract values for class 1
            if hasattr(ev, "values"):
                vals = ev.values
                if getattr(vals, "ndim", 0) == 3 and vals.shape[-1] > 1:
                    shap_values = vals[..., 1]
                else:
                    shap_values = vals
            else:
                # fallback: try to coerce to numpy
                shap_values = np.asarray(ev)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            # Last resort: return zeros
            shap_values = np.zeros((len(X), len(FEATURES)))

    return explainer, shap_values, X

def save_model(model, path="rf_model.joblib"):
    joblib.dump(model, path)

def load_model(path="rf_model.joblib"):
    return joblib.load(path)
