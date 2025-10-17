# app_react_style.py - Streamlit app with React-style interactivity
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from matplotlib import pyplot as plt
import os

from model_and_utils import train_rf, predict_grid
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

def get_simplified_explanation(model, point_data):
    """Get simplified feature explanation without SHAP"""
    features = ['vegetation', 'soil_moisture', 'distance_to_road', 'conflict_intensity', 'elevation']
    feature_importances = model.feature_importances_
    
    explanations = []
    for i, feature in enumerate(features):
        value = point_data[feature]
        importance = feature_importances[i]
        
        # Simple heuristic for contribution
        if feature == 'distance_to_road':
            # Closer to road = higher risk
            contribution = -importance * 0.1 * value
        else:
            # Higher value = higher risk
            contribution = importance * 0.1 * value
        
        explanations.append({
            'feature': feature,
            'value': value,
            'contribution': contribution
        })
    
    return explanations

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
st.markdown("## 🌍 Interactive Risk Assessment Map")
st.markdown("**Click on any marker to see detailed feature explanations!**")

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

# Add clickable markers with explanations
marker_sample = df_pred.sample(n=min(100, len(df_pred)), random_state=42)
for _, row in marker_sample.iterrows():
    color = "red" if row["mine"] == 1 else "green"
    risk_level = "High" if row["risk_proba"] > 0.7 else "Medium" if row["risk_proba"] > 0.4 else "Low"
    
    # Create popup with basic info
    popup_html = f"""
    <div style="font-family: Arial; width: 200px;">
        <h4>Risk Assessment</h4>
        <p><strong>Risk Level:</strong> {risk_level}</p>
        <p><strong>Probability:</strong> {row['risk_proba']:.1%}</p>
        <p><strong>Actual Mine:</strong> {'Yes' if row['mine'] == 1 else 'No'}</p>
        <p><strong>Click "Show Explanation" below for detailed analysis</strong></p>
    </div>
    """
    
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=max(3, min(8, row["risk_proba"] * 10)),
        color=color,
        fill=True,
        fillOpacity=0.7,
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(m)

# render map
st_data = st_folium(m, width=1000, height=600, key=f"map_{st.session_state.retrain_counter}_{radius}")

# -----------------------
# Feature Explanation Section
# -----------------------
st.markdown("---")
st.markdown("## 🔍 Feature Explanation Tool")

# Create two columns for the explanation interface
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Select a Point")
    
    # Create a dropdown to select points
    point_options = []
    for idx, row in marker_sample.iterrows():
        risk_level = "High" if row["risk_proba"] > 0.7 else "Medium" if row["risk_proba"] > 0.4 else "Low"
        point_options.append(f"Point {idx}: {risk_level} Risk ({row['risk_proba']:.1%})")
    
    selected_idx = st.selectbox("Choose a point to analyze:", range(len(point_options)), format_func=lambda x: point_options[x])
    
    if st.button("Show Feature Explanation"):
        selected_point = marker_sample.iloc[selected_idx]
        explanations = get_simplified_explanation(model, selected_point)
        
        # Store in session state for display
        st.session_state.selected_explanations = explanations
        st.session_state.selected_point = selected_point

with col2:
    st.markdown("### Feature Contributions")
    
    if 'selected_explanations' in st.session_state:
        selected_point = st.session_state.selected_point
        explanations = st.session_state.selected_explanations
        
        # Display point info
        st.info(f"**Location:** {selected_point['lat']:.4f}, {selected_point['lon']:.4f} | **Risk Probability:** {selected_point['risk_proba']:.1%}")
        
        # Display explanations
        for explanation in sorted(explanations, key=lambda x: abs(x['contribution']), reverse=True):
            feature_name = explanation['feature'].replace('_', ' ').title()
            value = explanation['value']
            contribution = explanation['contribution']
            
            # Color code the contribution
            if contribution > 0:
                color = "🔴"  # Red for risk-increasing
                impact = "increases"
            else:
                color = "🟢"  # Green for risk-decreasing
                impact = "decreases"
            
            st.markdown(f"""
            **{color} {feature_name}**
            - Value: {value:.3f}
            - Contribution: {contribution:+.3f} ({impact} risk)
            """)
    else:
        st.info("Select a point and click 'Show Feature Explanation' to see detailed analysis")

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
# Feature importance
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

# -----------------------
# Download CSV
# -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download current dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download CSV", csv, "current_dataset.csv", "text/csv")
