
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title('Crop Yield Prediction App')

# Load data
df = pd.read_csv("crop yield prediction.csv")

# Preprocessing
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Check target variable
if 'Yield' not in df.columns:
    st.error("Dataset must contain a 'Yield' column")
    st.stop()

if df['Yield'].dtype == 'object':
    st.error("Yield should be a numerical column for regression")
    st.stop()

# Feature selection
features = ['Crop', 'Precipitation (mm day-1)', 
            'Specific Humidity at 2 Meters (g/kg)',
            'Relative Humidity at 2 Meters (%)', 
            'Temperature at 2 Meters (C)']

X = df[features]
y = df['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
@st.cache_resource
def train_models():
    # Decision Tree
    dt_model = DecisionTreeRegressor(max_depth=30, min_samples_split=3, 
                                   min_samples_leaf=1, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    return dt_model, rf_model

dt_model, rf_model = train_models()

# Evaluate models
y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Display metrics
st.subheader('Model Performance')
col1, col2 = st.columns(2)
with col1:
    st.metric("Decision Tree MSE", f"{mse_dt:.4f}")
    st.metric("Decision Tree R²", f"{r2_dt:.4f}")
with col2:
    st.metric("Random Forest MSE", f"{mse_rf:.4f}")
    st.metric("Random Forest R²", f"{r2_rf:.4f}")

# Visualization function
def plot_results(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel("Actual Yield")
    ax.set_ylabel("Predicted Yield")
    ax.set_title(f'Actual vs Predicted ({model_name})')
    st.pyplot(fig)

# Show plots
st.subheader('Model Predictions Visualization')
plot_results(y_test, y_pred_dt, "Decision Tree")
plot_results(y_test, y_pred_rf, "Random Forest")
# Prediction interface
st.subheader('Make New Prediction')
with st.form("prediction_form"):
    st.write("Enter feature values:")
    crop = st.number_input("Crop (encoded)", min_value=0)
    precipitation = st.number_input("Precipitation (mm/day)", value=10.0)
    humidity = st.number_input("Specific Humidity (g/kg)", value=15.0)
    rel_humidity = st.number_input("Relative Humidity (%)", value=80.0)
    temp = st.number_input("Temperature (°C)", value=25.0)
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        new_data = pd.DataFrame([[crop, precipitation, humidity, rel_humidity, temp]],
                              columns=features)
        new_data_scaled = scaler.transform(new_data)
        
        dt_pred = dt_model.predict(new_data_scaled)[0]
        rf_pred = rf_model.predict(new_data_scaled)[0]
        
        st.success(f"Decision Tree Prediction: {dt_pred:.2f}")
        st.success(f"Random Forest Prediction: {rf_pred:.2f}")

# Show decision tree visualization
if st.checkbox("Show Decision Tree Structure"):
    plt.figure(figsize=(25, 20), dpi=300)
    plot_tree(dt_model, feature_names=features, filled=True, 
             rounded=True, fontsize=12)
    plt.title("Decision Tree Structure", fontsize=25)
    st.pyplot(plt.gcf())