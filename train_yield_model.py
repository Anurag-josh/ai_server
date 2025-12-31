import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump

# 🔢 Dummy training data (replace with real NDVI + yield data)
X = [
    [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.68, 0.6, 0.5, 0.4, 18.52, 73.85, 0],  # corn
    [0.2, 0.25, 0.3, 0.35, 0.4, 0.38, 0.36, 0.3, 0.25, 0.2, 18.6, 73.9, 1], # tomato
    [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.72, 0.68, 0.65, 0.6, 18.3, 73.5, 2], # onion
    [0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.94, 18.8, 73.3, 3]  # sugarcane
]
y = [4.5, 3.2, 5.0, 9.0]

X = np.array(X)
y = np.array(y)

# 📊 Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 💾 Save
dump(model, "yield_model.joblib")
print("✅ Model saved as yield_model.joblib")
