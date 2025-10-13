# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from matplotlib import pyplot as plt
import shap
import os

from model_and_utils import train_rf, predict_grid, incremental_update, explain_model_shap, save_model, load_model
from simulate_data import generate_synthetic_geodata

# -----------------------
# Config
# -----------------------
st.set_page_config(layout="wide", page_title="Dynamic Landmine Risk Intelligence System")

# -----------------------
# Helpers (cached)
# -----------------------
@st.cache_data
def load_data():
    """Generate synthetic data programmatically instead of loading from CSV"""
    df = generate_synthetic_geodata(n_points=1200, seed=42)
    return df

@st.cache_data
def get_trained_predictions(df_data, retrain_key=None):
    """
    Train model and return (model, metrics, feature_importances, df_pred).
    Caching keyed by retrain_key (increment when retraining) to avoid repeated recompute.
    """
    model, metrics, feature_importances = train_rf(df_data)
    df_pred = predict_grid(model, df_data)
    return model, metrics, feature_importances, df_pred

# -----------------------
# UI top
# -----------------------
st.title("🧭 Dynamic Landmine Risk Intelligence System")
st.caption("Developed by **Krish Jani** — Graduate Research Prototype for Geospatial Machine Learning")

# Load dataset
df = load_data()

# Initialize session state for user-added data
if "user_added_data" not in st.session_state:
    st.session_state.user_added_data = pd.DataFrame()

# Combine base data with user-added data
if not st.session_state.user_added_data.empty:
    df = pd.concat([df, st.session_state.user_added_data], ignore_index=True)

# Sidebar controls
st.sidebar.header("Controls")
radius = st.sidebar.slider("Heatmap radius", 8, 40, 18)
retrain_button = st.sidebar.button("Retrain model on current dataset")
add_point = st.sidebar.checkbox("Add a labelled point (simulate field report)")

# retrain counter to bust cache when user triggers retrain
if "retrain_counter" not in st.session_state:
    st.session_state.retrain_counter = 0
if retrain_button:
    st.session_state.retrain_counter += 1

# Train and predict (cached)
model, metrics, feature_importances, df_pred = get_trained_predictions(df, st.session_state.retrain_counter)

# Model summary on sidebar
st.sidebar.markdown("### Model Summary")
st.sidebar.write(f"CV AUC (mean ± std): {metrics.get('cv_auc_mean', float('nan')):.3f} ± {metrics.get('cv_auc_std', float('nan')):.3f}")
st.sidebar.write(f"Test AUC: {metrics.get('test_auc', float('nan')):.3f}")
st.sidebar.write(f"Test Accuracy: {metrics.get('test_accuracy', float('nan')):.3f}")
st.sidebar.write(f"Train size / Test size: {metrics.get('n_train')}/{metrics.get('n_test')}")

# -----------------------
# Dashboard overview
# -----------------------
st.markdown("## 📊 Dataset & Model Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total samples", len(df_pred))
col2.metric("High-risk (prob > 0.7)", int((df_pred["risk_proba"] > 0.7).sum()))
col3.metric("Average predicted risk", f"{df_pred['risk_proba'].mean():.3f}")

st.markdown("---")

# -----------------------
# Map & visualization
# -----------------------
st.markdown("## 🌍 Dynamic Heatmap Visualization")
mean_lat = df_pred["lat"].mean()
mean_lon = df_pred["lon"].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="CartoDB positron")

# Prepare heat data safely
df_heat = df_pred.dropna(subset=["lat", "lon", "risk_proba"]).copy()
df_heat = df_heat.astype({"lat": float, "lon": float, "risk_proba": float})
df_heat["lat"] = df_heat["lat"].round(4)
df_heat["lon"] = df_heat["lon"].round(4)
df_heat["risk_proba"] = df_heat["risk_proba"].clip(0, 1)

# sample to keep map responsive
np.random.seed(42)
df_heat_sample = df_heat.sample(n=min(800, len(df_heat)), random_state=42)
heat_data = df_heat_sample[["lat", "lon", "risk_proba"]].values.tolist()

HeatMap(
    heat_data,
    radius=radius,
    blur=20,
    max_zoom=12,
    min_opacity=0.3,
    gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
).add_to(m)

# overlay a small set of labeled markers
marker_sample = df_pred.sample(n=min(60, len(df_pred)), random_state=42)
for _, row in marker_sample.iterrows():
    color = "red" if row["mine"] == 1 else "green"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"risk:{row['risk_proba']:.2f}, label:{row['mine']}"
    ).add_to(m)

# render map. use retrain counter in key so map updates when retrained
st_data = st_folium(m, width=1000, height=600, key=f"map_{st.session_state.retrain_counter}_{radius}")

# -----------------------
# Add labeled point (simulate field report)
# -----------------------
if add_point:
    st.subheader("Add labelled point (simulate field observation)")
    with st.form("add_point_form"):
        lat = st.number_input("Latitude", value=float(mean_lat))
        lon = st.number_input("Longitude", value=float(mean_lon))
        vegetation = st.slider("Vegetation (0-1)", 0.0, 1.0, 0.4)
        soil_moisture = st.slider("Soil moisture (0-1)", 0.0, 1.0, 0.4)
        distance_to_road = st.number_input("Distance to road", value=1.0, step=0.1)
        conflict_intensity = st.selectbox("Conflict intensity (0-3)", [0, 1, 2, 3])
        elevation = st.number_input("Elevation (m)", 1200)
        label = st.radio("Label (mine present?)", options=[0, 1], index=0)
        submitted = st.form_submit_button("Add point and save")
        if submitted:
            new_row = pd.DataFrame([{
                "lon": lon, "lat": lat,
                "vegetation": vegetation,
                "soil_moisture": soil_moisture,
                "distance_to_road": distance_to_road,
                "conflict_intensity": conflict_intensity,
                "elevation": elevation,
                "mine": label
            }])
            # Add to session state instead of CSV file
            st.session_state.user_added_data = pd.concat([st.session_state.user_added_data, new_row], ignore_index=True)
            st.success("Added point to dataset. Click 'Retrain model on current dataset' to update predictions.")

# -----------------------
# Retrain action
# -----------------------
if retrain_button:
    st.success("Model retraining triggered — cache will refresh and the map will update.")

# -----------------------
# Risk distribution plot
# -----------------------
st.markdown("---")
st.markdown("### 🔎 Predicted risk distribution")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(df_pred["risk_proba"], bins=30, range=(0, 1))
ax.set_xlabel("Predicted risk probability")
ax.set_ylabel("Count")
st.pyplot(fig)

# -----------------------
# Feature importance & SHAP
# -----------------------
st.markdown("---")
st.markdown("### 🔍 Model Interpretability")

# feature importance bar
st.subheader("Top feature importances (Random Forest)")
fi = feature_importances.sort_values(ascending=True)
fig2, ax2 = plt.subplots(figsize=(6, 3))
fi.plot.barh(ax=ax2)
ax2.set_xlabel("Importance")
st.pyplot(fig2)

# SHAP controls
if st.checkbox("Show SHAP explanations (sample)"):
    st.write("Computing SHAP values (this can take a few seconds)...")
    sample = df.sample(n=min(200, len(df)), random_state=42)
    
    try:
        explainer, shap_values, X_sample = explain_model_shap(model, sample)
        st.success(f"SHAP values computed successfully! Shape: {shap_values.shape}")
        
        # Choose plot kind
        plot_kind = st.radio("SHAP plot type", ["Summary (beeswarm)", "Bar"], index=0)
        
        if plot_kind == "Bar":
            fig3 = plt.figure(figsize=(8, 4))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            st.pyplot(fig3)
        else:
            fig4 = plt.figure(figsize=(10, 5))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig4)
            
    except Exception as e:
        st.error(f"SHAP computation failed: {str(e)}")
        st.write("This might be due to insufficient data or model issues. Try retraining the model or adding more data points.")

# -----------------------
# Download CSV
# -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download current dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download CSV", csv, "current_dataset.csv", "text/csv")
