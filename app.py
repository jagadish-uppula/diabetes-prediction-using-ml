import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
import sys

# Add utils to path
sys.path.append('./utils')

try:
    from utils.data_loader import load_data
    from utils.visualization import (plot_distribution, 
                                    plot_correlation_matrix, 
                                    plot_feature_importance)
except ImportError as e:
    st.error(f"Import error: {e}")
    # Fallback functions
    def load_data(path):
        return pd.read_csv(path)
    
    def plot_distribution(data, feature):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data[feature].hist(ax=ax)
        return fig
    
    def plot_correlation_matrix(data):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = data.corr()
        im = ax.imshow(corr, cmap='coolwarm')
        plt.colorbar(im)
        return fig
    
    def plot_feature_importance(features, importances):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sorted_idx = np.argsort(importances)
        ax.barh(range(len(importances)), importances[sorted_idx])
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([features[i] for i in sorted_idx])
        return fig

# Set page config
st.set_page_config(
    page_title="AI Diabetes Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# Load custom CSS
def load_css():
    try:
        with open("static/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.markdown("""
        <style>
        .main { background-color: #f0f2f6; }
        </style>
        """, unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model(model_path='models/model.pkl'):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data with error handling
@st.cache_data
def load_data_cached(path='data/diabetes.csv'):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data
        return pd.DataFrame({
            'Pregnancies': [1, 2, 3],
            'Glucose': [100, 120, 140],
            'BloodPressure': [70, 80, 90],
            'SkinThickness': [20, 25, 30],
            'Insulin': [80, 100, 120],
            'BMI': [25.0, 27.0, 30.0],
            'DiabetesPedigree': [0.5, 0.6, 0.7],
            'Age': [30, 35, 40],
            'Outcome': [0, 1, 0]
        })

# Main app function
def main():
    load_css()
    model = load_model()
    data = load_data_cached()
    
    if model is None:
        st.warning("Model not loaded. Using demo mode.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose section", 
                               ["Home", "Data Exploration", "Diagnosis"])
    
    # Load images with error handling
    try:
        logo_img = Image.open("static/images/logo.png")
        diabetes_img = Image.open("static/images/diabetes_image.jpg")
    except:
        logo_img = None
        diabetes_img = None
    
    # Home page
    if app_mode == "Home":
        if logo_img:
            st.image(logo_img, width=200)
        st.title("AI-Powered Diabetes Diagnosis System")
        if diabetes_img:
            st.image(diabetes_img, use_column_width=True)
        st.markdown("""
        ## About This System
        This system helps healthcare professionals assess diabetes risk using machine learning.
        
        ### Features:
        - **Data Exploration**: Visualize diabetes dataset
        - **Patient Diagnosis**: Get instant risk assessment
        - **Model Insights**: Understand prediction factors
        """)
    
    # Data Exploration page
    elif app_mode == "Data Exploration":
        st.header("Data Exploration")
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(data)
        
        st.subheader("Data Visualization")
        feature = st.selectbox("Select feature", data.columns[:-1])
        st.pyplot(plot_distribution(data, feature))
        
        st.subheader("Correlation Matrix")
        st.pyplot(plot_correlation_matrix(data))
    
    # Diagnosis page
    elif app_mode == "Diagnosis":
        st.header("Patient Diabetes Risk Assessment")
        
        with st.form("diagnosis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Pregnancies", 0, 20, 1)
                glucose = st.number_input("Glucose (mg/dL)", 0, 300, 100)
                bp = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
                skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            
            with col2:
                insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0, 0.1)
                dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01)
                age = st.number_input("Age", 0, 120, 30)
            
            submitted = st.form_submit_button("Assess Risk")
            
            if submitted:
                if model is not None:
                    input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
                    prediction = model.predict(input_data)[0]
                    proba = model.predict_proba(input_data)[0]
                    
                    if prediction == 1:
                        st.error(f"High Risk ({proba[1]:.1%} probability)")
                        st.warning("Recommend consultation with a healthcare provider")
                    else:
                        st.success(f"Low Risk ({proba[0]:.1%} probability)")
                        st.info("Maintain healthy lifestyle with regular checkups")
                    
                    # Show feature importance
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Key Contributing Factors")
                        features = data.columns[:-1]
                        importances = model.feature_importances_
                        st.pyplot(plot_feature_importance(features, importances))
                else:
                    st.warning("Demo mode: Model not available")

if __name__ == "__main__":
    main()
