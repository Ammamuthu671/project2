import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Dry Bean Area Predictor",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Dry Bean Area Predictor App")
st.write("Enter the bean's physical characteristics below to predict its **Area** using your trained Linear Regression Model.")

# 2. Load the trained Pickle Model safely
@st.cache_resource
def load_model():
    with open('linear_regression_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'linear_regression_model.pkl' not found! Make sure the file is in the same directory as this script.")
    st.stop()

# 3. Define the exact feature names your model was trained on
feature_names = [
    'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 
    'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 
    'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

st.header("🔧 Input Bean Features")

# 4. Create Columns for a clean UI layout
col1, col2 = st.columns(2)

with col1:
    perimeter = st.number_input("Perimeter", min_value=0.0, max_value=2500.0, value=610.29, step=0.1)
    major_axis = st.number_input("Major Axis Length", min_value=0.0, max_value=1000.0, value=208.17, step=0.1)
    minor_axis = st.number_input("Minor Axis Length", min_value=0.0, max_value=500.0, value=173.88, step=0.1)
    aspect_ration = st.number_input("Aspect Ratio", min_value=0.0, max_value=5.0, value=1.19, step=0.01)
    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.54, step=0.01)
    convex_area = st.number_input("Convex Area", min_value=0.0, max_value=300000.0, value=28715.0, step=1.0)
    equiv_diameter = st.number_input("Equivalent Diameter", min_value=0.0, max_value=1000.0, value=190.14, step=0.1)
    extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.76, step=0.01)

with col2:
    solidity = st.number_input("Solidity", min_value=0.0, max_value=1.0, value=0.98, step=0.01)
    roundness = st.number_input("Roundness", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    compactness = st.number_input("Compactness", min_value=0.0, max_value=1.0, value=0.91, step=0.01)
    shape_factor1 = st.number_input("Shape Factor 1", min_value=0.0, max_value=0.1, value=0.007332, format="%.6f")
    shape_factor2 = st.number_input("Shape Factor 2", min_value=0.0, max_value=0.1, value=0.003147, format="%.6f")
    shape_factor3 = st.number_input("Shape Factor 3", min_value=0.0, max_value=1.0, value=0.834222, format="%.6f")
    shape_factor4 = st.number_input("Shape Factor 4", min_value=0.0, max_value=1.0, value=0.998724, format="%.6f")

# 5. Prediction Logic
st.markdown("---")
if st.button("🔮 Predict Area", type="primary"):
    # Group the values into a list matching the exact order
    input_data = [[
        perimeter, major_axis, minor_axis, aspect_ration, eccentricity,
        convex_area, equiv_diameter, extent, solidity, roundness,
        compactness, shape_factor1, shape_factor2, shape_factor3, shape_factor4
    ]]
    
    # Pack into a DataFrame so the model gets valid feature names
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Run the prediction
    prediction = model.predict(input_df)
    
    # Display Result
    st.success(f"### 🎉 Predicted Area: **{prediction[0]:,.2f} sq units**")