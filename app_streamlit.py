# app_streamlit.py
from matplotlib import pyplot as plt
import shap
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from model_and_utils import train_rf, predict_grid, incremental_update, explain_model_shap, save_model, load_model
import os

st.set_page_config(layout="wide", page_title="Dynamic Landmine Risk Prototype")

DATA_PATH = "synthetic_mine_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def train_and_predict(df_data, retrain_key=None):
    """Cache model training and predictions to prevent recomputation"""
    model, metrics = train_rf(df_data)
    df_pred = predict_grid(model, df_data)
    return model, metrics, df_pred

df = load_data()

st.sidebar.title("Controls")
radius = st.sidebar.slider("Heatmap radius", min_value=8, max_value=30, value=16)
retrain_button = st.sidebar.button("Retrain model on current dataset")
add_point = st.sidebar.checkbox("Add a labelled point (simulate field report)")

# Use session state to track retraining
if 'retrain_counter' not in st.session_state:
    st.session_state.retrain_counter = 0

if retrain_button:
    st.session_state.retrain_counter += 1

# train model and get predictions (cached)
model, metrics, df_pred = train_and_predict(df, st.session_state.retrain_counter)
st.sidebar.markdown(f"**Model AUC:** {metrics['auc']:.3f}  \n**Accuracy:** {metrics['accuracy']:.3f}")

# Main map
st.title("Dynamic Landmine Risk Heatmap — Demo")
st.markdown("This is a prototype: add labelled points to simulate field updates and retrain to see changes.")

# Build folium map centered on mean coordinates
mean_lat = df_pred["lat"].mean()
mean_lon = df_pred["lon"].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="CartoDB positron")

# Add heatmap layer using predicted probabilities (weighted)
# Clean and round data to avoid folium overflow issues
df_heat = df_pred.dropna(subset=["lat", "lon", "risk_proba"]).copy()
df_heat = df_heat.astype({"lat": float, "lon": float, "risk_proba": float})
df_heat["lat"] = df_heat["lat"].round(4)
df_heat["lon"] = df_heat["lon"].round(4)
df_heat["risk_proba"] = df_heat["risk_proba"].clip(0, 1)  # ensure valid range

# Use consistent sampling for stability
np.random.seed(42)
df_heat_sample = df_heat.sample(n=min(600, len(df_heat)), random_state=42)

# Construct heatmap data
heat_data = df_heat_sample[["lat", "lon", "risk_proba"]].values.tolist()

# Add single heatmap layer with optimized parameters
HeatMap(
    heat_data,
    radius=radius,
    blur=25,
    max_zoom=12,
    min_opacity=0.4,
    gradient={0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
).add_to(m)

# Add sample markers (colored by label) with consistent sampling
marker_sample = df_pred.sample(n=min(60, len(df_pred)), random_state=42)
for _, row in marker_sample.iterrows():
    color = "red" if row["mine"]==1 else "green"
    folium.CircleMarker(location=[row["lat"], row["lon"]],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"risk:{row['risk_proba']:.2f}, label:{row['mine']}").add_to(m)

# Render folium map in streamlit with a key to prevent unnecessary re-renders
st_data = st_folium(m, width=900, height=600, key=f"map_{st.session_state.retrain_counter}_{radius}")

# Simulate adding a labeled point
if add_point:
    st.subheader("Add a labeled point (simulate field observation)")
    with st.form("add_point_form"):
        lat = st.number_input("Latitude", value=float(mean_lat))
        lon = st.number_input("Longitude", value=float(mean_lon))
        vegetation = st.slider("Vegetation (0-1)", 0.0, 1.0, 0.4)
        soil_moisture = st.slider("Soil moisture (0-1)", 0.0, 1.0, 0.4)
        distance_to_road = st.number_input("Distance to road", value=1.0, step=0.1)
        conflict_intensity = st.selectbox("Conflict intensity (0-3)", [0,1,2,3])
        elevation = st.number_input("Elevation (m)", 1200)
        label = st.radio("Label (mine present?)", options=[0,1], index=0)
        submitted = st.form_submit_button("Add point and retrain")
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
            # Append to CSV (simulate recording)
            new_df = pd.read_csv(DATA_PATH)
            new_df = pd.concat([new_df, new_row], ignore_index=True)
            new_df.to_csv(DATA_PATH, index=False)
            st.success("Added point to dataset. Click 'Retrain model on current dataset' to update predictions.")

# Retrain model if requested
if retrain_button:
    df_current = pd.read_csv(DATA_PATH)
    model_new, metrics_new = train_rf(df_current)
    save_model(model_new, path="rf_model.joblib")
    st.success("Model retrained on the current dataset. The map will update automatically.")

# Provide ability to download current dataset
st.sidebar.markdown("---")
st.sidebar.markdown("Download current dataset")
csv = pd.read_csv(DATA_PATH).to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download CSV", csv, "current_dataset.csv", "text/csv")

# Explainability
if st.sidebar.checkbox("Show SHAP explanations (sample)"):
    st.write("Computing SHAP values (may take a few seconds)...")
    sample = pd.read_csv(DATA_PATH).sample(n=min(200, len(df)), random_state=42)
    feature_cols = ["vegetation", "soil_moisture", "distance_to_road", "conflict_intensity", "elevation"]
    X_sample = sample[feature_cols]

    # Use fast tree-specific explainer; compute class-1 SHAP values (not interactions)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values_list = explainer.shap_values(X_sample)
        # shap_values_list can be a list [class0, class1] for classifiers
        if isinstance(shap_values_list, list) and len(shap_values_list) > 1:
            shap_values = shap_values_list[1]
        else:
            shap_values = shap_values_list
    except Exception:
        # Fallback to generic explainer
        explainer, sv = explain_model_shap(model, sample)
        # sv might be an Explanation object; convert to array for class-1 if present
        try:
            if hasattr(sv, "values") and getattr(sv.values, "ndim", 0) == 3 and sv.values.shape[-1] > 1:
                shap_values = sv.values[..., 1]
            else:
                shap_values = sv.values if hasattr(sv, "values") else sv
        except Exception:
            shap_values = sv

    st.subheader("SHAP Feature Importance")
    # Controls for plot appearance
    plot_kind = st.radio("Plot type", ["Bar", "Beeswarm"], index=0, horizontal=True)
    max_display = st.slider("Max features", min_value=3, max_value=len(feature_cols), value=5)

    # Create the plot with improved sizing and margins to prevent clipping
    plt.close('all')
    # Slightly larger default font for readability
    plt.rcParams.update({"font.size": 11})
    fig = plt.figure(figsize=(12, 7))
    if plot_kind == "Bar":
        shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_display, show=False, plot_size=(12, 7))
    else:
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False, plot_size=(12, 7))
    # Adjust margins to avoid cutting labels
    plt.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.93)
    # Tight bounding box + container width to avoid overflow/cropping
    st.pyplot(plt.gcf(), clear_figure=True, bbox_inches='tight', pad_inches=0.2, use_container_width=True)
