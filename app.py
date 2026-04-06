# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# 1. Load the trained model (ensure 'car_price_model.pkl' is in the same folder)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

model = load_model()

# Extract the preprocessor and encoder from the pipeline
preprocessor = model.named_steps['preprocessor']
encoder = preprocessor.named_transformers_['cat']

# Get the categories learned during training for each categorical feature
cat_columns = encoder.feature_names_in_  # list of categorical column names
categories = encoder.categories_          # list of arrays, one per categorical column

# ------------------------------------------------------------
# 2. Feature list (must match training order)
# ------------------------------------------------------------
feature_cols_enhanced = [
    'production_year', 'levy', 'mileage', 'cylinders', 'airbags', 'doors',
    'manufacturer', 'model', 'fuel_type', 'category', 'leather_interior',
    'gear_box_type', 'drive_wheels', 'wheel', 'color', 'engine_volume',
    'car_age', 'age_group', 'mileage_group', 'engine_per_cylinder', 'production_year_squared'
]

# ------------------------------------------------------------
# 3. Helper: apply the same feature engineering as in the notebook
# ------------------------------------------------------------
def apply_feature_engineering(df):
    current_year = 2024   # same as in training
    df['car_age'] = current_year - df['production_year']
    
    # Age group bins
    df['age_group'] = pd.cut(df['car_age'],
                             bins=[0, 5, 10, 15, 100],
                             labels=['New', 'Recent', 'Mid-age', 'Old'])
    
    # Mileage group bins
    df['mileage_group'] = pd.cut(df['mileage'],
                                 bins=[0, 50000, 100000, 150000, 1000000],
                                 labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Engine efficiency
    df['engine_per_cylinder'] = df['engine_volume'] / df['cylinders']
    
    # Polynomial feature
    df['production_year_squared'] = df['production_year'] ** 2
    
    return df

# ------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction")
st.markdown("Enter the car details below to get an estimated price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        prod_year = st.number_input("Production Year", min_value=1990, max_value=2024, value=2018, step=1)
        levy = st.number_input("Levy (tax)", min_value=0.0, value=100.0, step=50.0)
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000, step=10000)
        cylinders = st.selectbox("Cylinders", [2, 3, 4, 5, 6, 8, 10, 12], index=2)
        airbags = st.slider("Airbags", 0, 16, 4)
        doors = st.selectbox("Doors", [2, 3, 4, 5], index=2)
        engine_vol = st.number_input("Engine Volume (liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    
    with col2:
        # Build dropdowns using categories from the trained encoder
        manufacturer = st.selectbox("Manufacturer", categories[cat_columns.tolist().index('manufacturer')])
        model_cat = st.selectbox("Model", categories[cat_columns.tolist().index('model')])
        fuel_type = st.selectbox("Fuel Type", categories[cat_columns.tolist().index('fuel_type')])
        category = st.selectbox("Category", categories[cat_columns.tolist().index('category')])
        leather_interior = st.selectbox("Leather Interior", categories[cat_columns.tolist().index('leather_interior')])
        gear_box_type = st.selectbox("Gear Box Type", categories[cat_columns.tolist().index('gear_box_type')])
        drive_wheels = st.selectbox("Drive Wheels", categories[cat_columns.tolist().index('drive_wheels')])
        wheel = st.selectbox("Wheel (steering side)", categories[cat_columns.tolist().index('wheel')])
        color = st.selectbox("Color", categories[cat_columns.tolist().index('color')])
    
    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Build a DataFrame with the user inputs (one row)
    input_data = pd.DataFrame([{
        'production_year': prod_year,
        'levy': levy,
        'mileage': mileage,
        'cylinders': cylinders,
        'airbags': airbags,
        'doors': doors,
        'manufacturer': manufacturer,
        'model': model_cat,
        'fuel_type': fuel_type,
        'category': category,
        'leather_interior': leather_interior,
        'gear_box_type': gear_box_type,
        'drive_wheels': drive_wheels,
        'wheel': wheel,
        'color': color,
        'engine_volume': engine_vol
    }])
    
    # Apply the same feature engineering
    input_data = apply_feature_engineering(input_data)
    
    # Ensure columns are in the exact order expected by the model
    input_data = input_data[feature_cols_enhanced]
    
    # Predict
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 **Estimated Price:** ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
    
    # Optional: show a note about confidence
    st.caption("Prediction is based on a Random Forest model trained on 19,237 cars. Actual market prices may vary.")
