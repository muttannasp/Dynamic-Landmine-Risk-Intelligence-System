import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMapEvents } from 'react-leaflet';
import { LatLng } from 'leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import './App.css';

// Fix for default markers in react-leaflet
import L from 'leaflet';
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

interface DataPoint {
  lat: number;
  lon: number;
  vegetation: number;
  soil_moisture: number;
  distance_to_road: number;
  conflict_intensity: number;
  elevation: number;
  mine: number;
  risk_proba: number;
}

interface ShapExplanation {
  feature: string;
  value: number;
  contribution: number;
}

interface PointExplanation {
  coordinates: { lat: number; lon: number };
  risk_probability: number;
  features: { [key: string]: number };
  shap_explanations: ShapExplanation[];
}

interface ModelMetrics {
  test_auc: number;
  test_accuracy: number;
  cv_auc_mean: number;
  cv_auc_std: number;
  n_train: number;
  n_test: number;
}

const API_BASE = 'http://localhost:8000';

function App() {
  const [data, setData] = useState<DataPoint[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<PointExplanation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
    fetchMetrics();
  }, []);

  const fetchData = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/data`);
      setData(response.data);
    } catch (err) {
      setError('Failed to fetch data');
      console.error(err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/metrics`);
      setMetrics(response.data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePointClick = async (lat: number, lon: number) => {
    try {
      const response = await axios.post(`${API_BASE}/api/explain`, {
        lat,
        lon
      });
      setSelectedPoint(response.data);
    } catch (err) {
      console.error('Failed to get explanation:', err);
    }
  };

  const getMarkerColor = (riskProba: number, mine: number) => {
    if (mine === 1) return '#ff0000'; // Red for actual mines
    if (riskProba > 0.7) return '#ff6b6b'; // Light red for high risk
    if (riskProba > 0.4) return '#ffa500'; // Orange for medium risk
    return '#00ff00'; // Green for low risk
  };

  const getMarkerRadius = (riskProba: number) => {
    return Math.max(3, Math.min(8, riskProba * 10));
  };

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading Dynamic Landmine Risk Intelligence System...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🧭 Dynamic Landmine Risk Intelligence System</h1>
        <p>Developed by <strong>Krish Jani</strong> — Graduate Research Prototype for Geospatial Machine Learning</p>
      </header>

      <div className="main-content">
        <div className="sidebar">
          <div className="metrics-section">
            <h3>📊 Model Performance</h3>
            {metrics && (
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Test AUC:</span>
                  <span className="metric-value">{metrics.test_auc.toFixed(3)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Test Accuracy:</span>
                  <span className="metric-value">{(metrics.test_accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="metric">
                  <span className="metric-label">CV AUC:</span>
                  <span className="metric-value">{metrics.cv_auc_mean.toFixed(3)} ± {metrics.cv_auc_std.toFixed(3)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Train/Test:</span>
                  <span className="metric-value">{metrics.n_train}/{metrics.n_test}</span>
                </div>
              </div>
            )}
          </div>

          <div className="legend-section">
            <h3>🎨 Map Legend</h3>
            <div className="legend">
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#ff0000' }}></div>
                <span>Actual Mine (Red)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#ff6b6b' }}></div>
                <span>High Risk (>70%)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#ffa500' }}></div>
                <span>Medium Risk (40-70%)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#00ff00' }}></div>
                <span>Low Risk (<40%)</span>
              </div>
            </div>
          </div>

          <div className="instructions">
            <h3>💡 Instructions</h3>
            <p>Click on any marker to see detailed SHAP explanations showing which factors contributed to the risk prediction.</p>
          </div>
        </div>

        <div className="map-container">
          <MapContainer
            center={[3.65, -76.4]}
            zoom={12}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            />
            
            {data.map((point, index) => (
              <CircleMarker
                key={index}
                center={[point.lat, point.lon]}
                radius={getMarkerRadius(point.risk_proba)}
                color={getMarkerColor(point.risk_proba, point.mine)}
                fillColor={getMarkerColor(point.risk_proba, point.mine)}
                fillOpacity={0.7}
                weight={2}
                eventHandlers={{
                  click: () => handlePointClick(point.lat, point.lon)
                }}
              >
                <Popup>
                  <div>
                    <strong>Risk Probability:</strong> {(point.risk_proba * 100).toFixed(1)}%<br/>
                    <strong>Actual Mine:</strong> {point.mine ? 'Yes' : 'No'}<br/>
                    <strong>Click for detailed explanation</strong>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>
      </div>

      {selectedPoint && (
        <div className="explanation-modal">
          <div className="explanation-content">
            <div className="explanation-header">
              <h2>🔍 SHAP Explanation</h2>
              <button 
                className="close-button"
                onClick={() => setSelectedPoint(null)}
              >
                ×
              </button>
            </div>
            
            <div className="explanation-body">
              <div className="coordinates">
                <strong>Location:</strong> {selectedPoint.coordinates.lat.toFixed(4)}, {selectedPoint.coordinates.lon.toFixed(4)}
              </div>
              
              <div className="risk-probability">
                <strong>Predicted Risk:</strong> {(selectedPoint.risk_probability * 100).toFixed(1)}%
              </div>

              <div className="feature-contributions">
                <h3>Feature Contributions to Risk Prediction:</h3>
                {selectedPoint.shap_explanations
                  .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                  .map((explanation, index) => (
                    <div key={index} className="contribution-item">
                      <div className="feature-name">{explanation.feature.replace('_', ' ').toUpperCase()}</div>
                      <div className="feature-value">Value: {explanation.value.toFixed(3)}</div>
                      <div className={`contribution ${explanation.contribution > 0 ? 'positive' : 'negative'}`}>
                        Contribution: {explanation.contribution > 0 ? '+' : ''}{explanation.contribution.toFixed(3)}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;