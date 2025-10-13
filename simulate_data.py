# simulate_data.py
import numpy as np
import pandas as pd

def generate_synthetic_geodata(n_points=800, seed=42, bbox=(-76.8, 3.3, -76.0, 4.0)):
    """
    Generate synthetic geospatial dataset in a bounding box.
    bbox = (lon_min, lat_min, lon_max, lat_max)
    """
    np.random.seed(seed)
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = np.random.uniform(lon_min, lon_max, size=n_points)
    lats = np.random.uniform(lat_min, lat_max, size=n_points)

    # simulate features
    vegetation = np.random.beta(2, 5, size=n_points)         # 0..1
    soil_moisture = np.random.beta(2, 3, size=n_points)      # 0..1
    distance_to_road = np.random.exponential(scale=1.0, size=n_points) # km-ish
    conflict_intensity = np.random.choice([0,1,2,3], size=n_points, p=[0.6, 0.2, 0.15, 0.05])
    elevation = np.random.normal(loc=1200, scale=150, size=n_points)  # meters

    # create an underlying risk function (nonlinear)
    risk_score = (
        2.5 * vegetation +
        1.8 * soil_moisture +
        -0.6 * np.log1p(distance_to_road) +
        0.9 * (conflict_intensity / 3.0) +
        0.001 * (elevation - 1200)
    )

    # convert to probability with sigmoid and add noise
    prob = 1 / (1 + np.exp(- (risk_score - 1.2)))
    prob = 0.65 * prob + 0.1 * np.random.rand(n_points)  # add randomness

    labels = (np.random.rand(n_points) < prob).astype(int)

    df = pd.DataFrame({
        "lon": lons,
        "lat": lats,
        "vegetation": vegetation,
        "soil_moisture": soil_moisture,
        "distance_to_road": distance_to_road,
        "conflict_intensity": conflict_intensity,
        "elevation": elevation,
        "mine": labels
    })
    return df

if __name__ == "__main__":
    df = generate_synthetic_geodata(n_points=1200)
    df.to_csv("synthetic_mine_data.csv", index=False)
    print("Saved synthetic_mine_data.csv (1200 rows)")
