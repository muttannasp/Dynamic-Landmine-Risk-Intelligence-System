# app_streamlit.py
import shap
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from model_and_utils import train_rf, predict_grid, incremental_update, explain_model_shap, save_model, load_model
import os

st.write("✅ App loaded successfully")

st.set_page_config(layout="wide", page_title="Dynamic Landmine Risk Prototype")

DATA_PATH = "synthetic_mine_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

st.sidebar.title("Controls")
radius = st.sidebar.slider("Heatmap radius", min_value=8, max_value=30, value=16)
retrain_button = st.sidebar.button("Retrain model on current dataset")
add_point = st.sidebar.checkbox("Add a labelled point (simulate field report)")

# train initial model
model, metrics = train_rf(df)
st.sidebar.markdown(f"**Initial model AUC:** {metrics['auc']:.3f}  \n**Accuracy:** {metrics['accuracy']:.3f}")

# Main map
st.title("Dynamic Landmine Risk Heatmap — Demo")
st.markdown("This is a prototype: add labelled points to simulate field updates and retrain to see changes.")

# Predict risk for all points
df_pred = predict_grid(model, df)

# Build folium map centered on mean coordinates
mean_lat = df_pred["lat"].mean()
mean_lon = df_pred["lon"].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="CartoDB positron")

# Add heatmap layer using predicted probabilities (weighted)
heat_data = list(zip(df_pred["lat"], df_pred["lon"], df_pred["risk_proba"]))
HeatMap(heat_data, radius=radius, blur=15, max_zoom=13).add_to(m)

# Add sample markers (colored by label)
for _, row in df_pred.sample(n=min(80, len(df_pred)), random_state=42).iterrows():
    color = "red" if row["mine"]==1 else "green"
    folium.CircleMarker(location=[row["lat"], row["lon"]],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"risk:{row['risk_proba']:.2f}, label:{row['mine']}").add_to(m)

# Render folium map in streamlit
st_data = st_folium(m, width=900, height=600)

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
    st.sidebar.markdown(f"**Retrained AUC:** {metrics_new['auc']:.3f}  \n**Accuracy:** {metrics_new['accuracy']:.3f}")
    st.success("Model retrained on the current dataset. Refresh the page to see updated map.")

# Provide ability to download current dataset
st.sidebar.markdown("---")
st.sidebar.markdown("Download current dataset")
csv = pd.read_csv(DATA_PATH).to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download CSV", csv, "current_dataset.csv", "text/csv")

# Explainability
if st.sidebar.checkbox("Show SHAP explanations (sample)"):
    st.write("Computing SHAP values (may take a few seconds)...")
    sample = pd.read_csv(DATA_PATH).sample(n=min(200, len(df)))
    explainer, shap_values = explain_model_shap(model, sample)
    st.write("SHAP summary (feature impact on model output):")
    import matplotlib.pyplot as plt
    fig = shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches="tight")
