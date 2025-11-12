import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from crop_info import CropInfo
from utils import validate_input, get_feature_ranges
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Professional Theme CSS
st.markdown("""
<style>
/* Main background and styling */
.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #1C1E26 50%, #262A36 100%);
    color: #FAFAFA;
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #00D4AA 0%, #00B894 100%);
}

/* Headers styling */
h1 {
    color: #FAFAFA !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    border-bottom: 3px solid #00D4AA;
    padding-bottom: 10px;
}

h2, h3 {
    color: #00D4AA !important;
    margin-top: 30px;
}

/* Metric cards styling */
div[data-testid="metric-container"] {
    background: linear-gradient(45deg, #1E2329, #2D3748);
    border: 1px solid #00D4AA;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 212, 170, 0.2);
    transition: transform 0.3s ease;
    color: #FAFAFA;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 212, 170, 0.3);
    background: linear-gradient(45deg, #2D3748, #3A4A5C);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #00D4AA, #00B894) !important;
    color: #0E1117 !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 10px 30px !important;
    font-weight: bold !important;
    box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(0, 212, 170, 0.5) !important;
    background: linear-gradient(45deg, #00E5BB, #00D4AA) !important;
}

/* Form styling */
.stForm {
    background: linear-gradient(135deg, #1E2329, #2D3748);
    border-radius: 20px;
    padding: 25px;
    border: 2px solid #3A4A5C;
    box-shadow: 0 8px 16px rgba(0, 212, 170, 0.1);
}

/* Selectbox and input styling */
.stSelectbox > div > div {
    border-radius: 10px;
    border: 2px solid #3A4A5C;
    background-color: #1E2329;
    color: #FAFAFA;
}

.stNumberInput > div > div > input {
    border-radius: 10px;
    border: 2px solid #3A4A5C;
    background-color: #1E2329;
    color: #FAFAFA;
}

/* Success, warning, error message styling */
.stSuccess {
    background: linear-gradient(45deg, #065F46, #047857);
    border-left: 5px solid #00D4AA;
    border-radius: 10px;
    color: #FAFAFA;
}

.stWarning {
    background: linear-gradient(45deg, #92400E, #B45309);
    border-left: 5px solid #F59E0B;
    border-radius: 10px;
    color: #FAFAFA;
}

.stError {
    background: linear-gradient(45deg, #991B1B, #B91C1C);
    border-left: 5px solid #EF4444;
    border-radius: 10px;
    color: #FAFAFA;
}

/* Progress bar styling */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00D4AA, #00E5BB);
    border-radius: 10px;
}

/* Data frame styling */
.dataframe {
    border-radius: 10px;
    border: 1px solid #3A4A5C;
    box-shadow: 0 2px 4px rgba(0, 212, 170, 0.1);
    background-color: #1E2329;
    color: #FAFAFA;
}

/* Custom container styling */
.custom-container {
    background: linear-gradient(135deg, #1E2329, #2D3748);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #3A4A5C;
    box-shadow: 0 4px 8px rgba(0, 212, 170, 0.1);
}

/* Sidebar text color */
.css-1d391kg .stMarkdown {
    color: #0E1117 !important;
    font-weight: 600;
}

.css-1d391kg .stSelectbox label {
    color: #0E1117 !important;
    font-weight: 600;
}

/* Text color overrides */
.stMarkdown, .stText, p, li, div {
    color: #FAFAFA !important;
}

/* Plot background styling */
.js-plotly-plot {
    background-color: #1E2329 !important;
    border-radius: 10px;
}

/* Input field focus styling */
.stNumberInput > div > div > input:focus {
    border-color: #00D4AA !important;
    box-shadow: 0 0 0 1px #00D4AA !important;
}

.stSelectbox > div > div:focus-within {
    border-color: #00D4AA !important;
    box-shadow: 0 0 0 1px #00D4AA !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1E2329;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #00D4AA;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #00E5BB;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None

def load_and_train_model():
    """Load data and train the model"""
    try:
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Load and preprocess data
        with st.spinner("Loading and preprocessing data..."):
            X_train, X_test, y_train, y_test, label_encoder = data_processor.load_and_preprocess_data()
        
        # Train model
        model_trainer = ModelTrainer()
        with st.spinner("Training machine learning model..."):
            model, metrics = model_trainer.train_model(X_train, X_test, y_train, y_test)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.data_processor = data_processor
        st.session_state.label_encoder = label_encoder
        st.session_state.metrics = metrics
        st.session_state.model_trained = True
        
        return True
    except Exception as e:
        st.error(f"Error loading/training model: {str(e)}")
        return False

def show_data_visualization():
    """Display data visualization"""
    if st.session_state.data_processor is None:
        return
    
    data = st.session_state.data_processor.get_raw_data()
    
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)
    with col3:
        st.metric("Crop Types", data['label'].nunique())
    with col4:
        st.metric("Data Quality", "100%" if data.isnull().sum().sum() == 0 else "Has Missing Values")
    
    # Feature distribution plots
    st.subheader("🔍 Feature Distributions")
    
    # Select features for visualization
    numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Create subplot for feature distributions
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=numeric_features,
        specs=[[{"secondary_y": False}]*4, [{"secondary_y": False}]*4]
    )
    
    for i, feature in enumerate(numeric_features):
        row = i // 4 + 1
        col = i % 4 + 1
        
        fig.add_trace(
            go.Histogram(x=data[feature], name=feature, showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Distribution of Agricultural Features")
    st.plotly_chart(fig, width='stretch')
    
    # Crop distribution
    st.subheader("🌱 Crop Distribution")
    crop_counts = data['label'].value_counts()
    
    fig_pie = px.pie(
        values=crop_counts.values,
        names=crop_counts.index,
        title="Distribution of Crops in Dataset"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, width='stretch')
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlation Matrix")
    correlation_data = data[numeric_features].corr()
    
    fig_heatmap = px.imshow(
        correlation_data,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Agricultural Features",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig_heatmap, width='stretch')

def show_model_performance():
    """Display model performance metrics"""
    if not st.session_state.model_trained:
        return
    
    st.subheader("🎯 Model Performance")
    
    metrics = st.session_state.metrics
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    
    # Feature importance
    if hasattr(st.session_state.model, 'feature_importances_'):
        st.subheader("📈 Feature Importance")
        feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': st.session_state.model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance in Crop Prediction"
        )
        st.plotly_chart(fig_importance, width='stretch')

def make_prediction(input_features):
    """Make crop prediction"""
    if not st.session_state.model_trained:
        return None, None
    
    try:
        # Prepare input for prediction
        input_array = np.array([input_features])
        
        # Scale the input
        scaled_input = st.session_state.data_processor.scaler.transform(input_array)
        
        # Make prediction
        prediction = st.session_state.model.predict(scaled_input)
        prediction_proba = st.session_state.model.predict_proba(scaled_input)
        
        # Decode prediction
        predicted_crop = st.session_state.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(prediction_proba) * 100
        
        return predicted_crop, confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    """Main application"""
    # Enhanced Header with dark professional styling
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #00D4AA, #00B894); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 16px rgba(0, 212, 170, 0.3);">
        <h1 style="color: #0E1117; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); border: none; font-weight: 700;">
            🌾 AI-Powered Crop Recommendation System
        </h1>
        <p style="color: #0E1117; font-size: 1.2em; margin: 10px 0 0 0; font-weight: 600;">
            Optimize your crop selection based on soil and environmental conditions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Analysis", "🎯 Model Performance", "🔮 Crop Prediction"]
    )
    
    # Load model if not already loaded
    if not st.session_state.model_trained:
        if st.sidebar.button("🚀 Initialize System", type="primary"):
            if load_and_train_model():
                st.success("✅ System initialized successfully!")
                st.rerun()
        else:
            st.warning("⚠️ Please initialize the system using the sidebar button.")
            st.stop()
    
    # Page routing
    if page == "🏠 Home":
        st.markdown("""
        <div class="custom-container">
            <h2 style="text-align: center; color: #00D4AA; margin-bottom: 20px;">
                🌿 Welcome to the AI Crop Recommendation System
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # What This System Does section
        st.markdown("""
        <div class="custom-container">
            <h3 style="color: #00D4AA;">🎯 What This System Does</h3>
            <p style="font-size: 1.1em; line-height: 1.6; color: #FAFAFA;">
                This intelligent system analyzes soil and environmental conditions to recommend 
                the most suitable crop for your farm using advanced machine learning algorithms.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input Parameters section
        st.markdown("""
        <div class="custom-container">
            <h3 style="color: #00D4AA;">📋 Required Input Parameters</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px;">
                <div style="background: #2D3748; padding: 15px; border-radius: 10px; border-left: 4px solid #00D4AA; color: #FAFAFA;">
                    <strong>🌱 N (Nitrogen):</strong> Nitrogen content in soil
                </div>
                <div style="background: #2D3748; padding: 15px; border-radius: 10px; border-left: 4px solid #00E5BB; color: #FAFAFA;">
                    <strong>🧪 P (Phosphorus):</strong> Phosphorus content in soil
                </div>
                <div style="background: #2D3748; padding: 15px; border-radius: 10px; border-left: 4px solid #00D4AA; color: #FAFAFA;">
                    <strong>⚗️ K (Potassium):</strong> Potassium content in soil
                </div>
                <div style="background: #2D3748; padding: 15px; border-radius: 10px; border-left: 4px solid #00E5BB; color: #FAFAFA;">
                    <strong>🌡️ Temperature:</strong> Average temperature (°C)
                </div>
                <div style="background: #2D3748; padding: 15px; border-radius: 10px; border-left: 4px solid #00D4AA; color: #FAFAFA;">
                    <strong>💧 Humidity:</strong> Relative humidity (%)
                </div>
                <div style="background: #2D3748; padding: 15px; border-radius: 10px; border-left: 4px solid #00E5BB; color: #FAFAFA;">
                    <strong>⚖️ pH Level:</strong> Soil pH level
                </div>
                <div style="background: #2D3748; padding: 15px; border-radius: 10px; border-left: 4px solid #00D4AA; color: #FAFAFA;">
                    <strong>🌧️ Rainfall:</strong> Annual rainfall (mm)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Supported Crops section
        st.markdown("""
        <div class="custom-container">
            <h3 style="color: #00D4AA;">🌱 Supported Crops (22 types)</h3>
            <div style="background: linear-gradient(135deg, #2D3748, #3A4A5C); padding: 20px; border-radius: 15px; border: 2px solid #00D4AA;">
                <p style="font-size: 1.1em; line-height: 1.8; color: #FAFAFA; text-align: center;">
                    🌾 Rice • 🌽 Maize • 🫘 Chickpea • 🫘 Kidneybeans • 🫘 Pigeonpeas • 🫘 Mothbeans <br>
                    🫘 Mungbean • 🫘 Blackgram • 🫘 Lentil • 🍎 Pomegranate • 🍌 Banana • 🥭 Mango <br>
                    🍇 Grapes • 🍉 Watermelon • 🍈 Muskmelon • 🍎 Apple • 🍊 Orange • 🥭 Papaya <br>
                    🥥 Coconut • 🌱 Cotton • 🌾 Jute • ☕ Coffee
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # How to Use section
        st.markdown("""
        <div class="custom-container">
            <h3 style="color: #00D4AA;">🚀 How to Use</h3>
            <div style="background: linear-gradient(135deg, #2D3748, #3A4A5C); padding: 20px; border-radius: 15px; border: 2px solid #3A4A5C;">
                <ol style="font-size: 1.1em; line-height: 2; color: #FAFAFA;">
                    <li><strong>Navigate</strong> to "🔮 Crop Prediction" in the sidebar</li>
                    <li><strong>Enter</strong> your soil and environmental parameters</li>
                    <li><strong>Get</strong> instant crop recommendations with confidence scores</li>
                    <li><strong>Explore</strong> additional crop information and farming tips</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.model_trained:
            st.markdown("""
            <div style="text-align: center; margin-top: 30px;">
                <div style="background: linear-gradient(45deg, #065F46, #047857); padding: 20px; border-radius: 15px; border: 2px solid #00D4AA; box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3);">
                    <h3 style="color: #00D4AA; margin: 0;">✅ System is ready for predictions!</h3>
                    <p style="color: #FAFAFA; margin: 10px 0 0 0;">Your AI model is trained and ready to help you choose the perfect crop.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "📊 Data Analysis":
        st.header("Data Analysis & Visualization")
        show_data_visualization()
    
    elif page == "🎯 Model Performance":
        st.header("Model Performance Metrics")
        show_model_performance()
    
    elif page == "🔮 Crop Prediction":
        st.header("🔮 Crop Prediction")
        
        # Get feature ranges for validation
        ranges = get_feature_ranges()
        
        # Input form
        with st.form("prediction_form"):
            st.subheader("📝 Enter Soil and Environmental Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n = st.number_input(
                    "Nitrogen (N)",
                    min_value=float(ranges['N']['min']),
                    max_value=float(ranges['N']['max']),
                    value=float(ranges['N']['default']),
                    help="Nitrogen content in soil"
                )
                
                p = st.number_input(
                    "Phosphorus (P)",
                    min_value=float(ranges['P']['min']),
                    max_value=float(ranges['P']['max']),
                    value=float(ranges['P']['default']),
                    help="Phosphorus content in soil"
                )
                
                k = st.number_input(
                    "Potassium (K)",
                    min_value=float(ranges['K']['min']),
                    max_value=float(ranges['K']['max']),
                    value=float(ranges['K']['default']),
                    help="Potassium content in soil"
                )
                
                temperature = st.number_input(
                    "Temperature (°C)",
                    min_value=float(ranges['temperature']['min']),
                    max_value=float(ranges['temperature']['max']),
                    value=float(ranges['temperature']['default']),
                    help="Average temperature in Celsius"
                )
            
            with col2:
                humidity = st.number_input(
                    "Humidity (%)",
                    min_value=float(ranges['humidity']['min']),
                    max_value=float(ranges['humidity']['max']),
                    value=float(ranges['humidity']['default']),
                    help="Relative humidity percentage"
                )
                
                ph = st.number_input(
                    "pH Level",
                    min_value=float(ranges['ph']['min']),
                    max_value=float(ranges['ph']['max']),
                    value=float(ranges['ph']['default']),
                    help="Soil pH level (0-14 scale)"
                )
                
                rainfall = st.number_input(
                    "Rainfall (mm)",
                    min_value=float(ranges['rainfall']['min']),
                    max_value=float(ranges['rainfall']['max']),
                    value=float(ranges['rainfall']['default']),
                    help="Annual rainfall in millimeters"
                )
            
            submitted = st.form_submit_button("🚀 Get Crop Recommendation", type="primary")
        
        if submitted:
            # Validate inputs
            input_features = [n, p, k, temperature, humidity, ph, rainfall]
            validation_result = validate_input(input_features)
            
            if validation_result['valid']:
                # Make prediction
                predicted_crop, confidence = make_prediction(input_features)
                
                if predicted_crop and confidence:
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Main result
                    st.subheader("🌾 Recommended Crop")
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### **{predicted_crop.title()}**")
                        st.markdown(f"**Confidence Score:** {confidence:.1f}%")
                        
                        # Confidence indicator
                        if confidence >= 80:
                            st.success("🟢 High Confidence - Excellent match!")
                        elif confidence >= 60:
                            st.warning("🟡 Medium Confidence - Good match")
                        else:
                            st.info("🔵 Low Confidence - Consider alternatives")
                    
                    with col2:
                        # Progress bar for confidence
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.progress(confidence / 100)
                    
                    # Crop information
                    crop_info = CropInfo()
                    if predicted_crop in crop_info.crop_database:
                        st.subheader("🌱 Crop Information")
                        info = crop_info.crop_database[predicted_crop]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Category:** {info['category']}")
                            st.markdown(f"**Season:** {info['season']}")
                        with col2:
                            st.markdown(f"**Water Requirement:** {info['water_requirement']}")
                            st.markdown(f"**Soil Type:** {info['soil_type']}")
                        with col3:
                            st.markdown(f"**Climate:** {info['climate']}")
                            st.markdown(f"**Harvest Time:** {info['harvest_time']}")
                        
                        # Farming tips
                        st.subheader("💡 Farming Tips")
                        for tip in info['tips']:
                            st.markdown(f"• {tip}")
                    
                    # Input summary
                    st.subheader("📋 Input Summary")
                    input_df = pd.DataFrame({
                        'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                                    'Temperature (°C)', 'Humidity (%)', 'pH Level', 'Rainfall (mm)'],
                        'Value': input_features
                    })
                    st.dataframe(input_df, width='stretch')
                    
            else:
                st.error(f"❌ {validation_result['message']}")

if __name__ == "__main__":
    main()
