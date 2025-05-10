import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Survival Prediction",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to apply the pink theme with consistent input field styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --pink-light: #F9F5F6;
        --pink-medium-light: #F8E8EE;
        --pink-medium: #FDCEDF;
        --pink-dark: #F2BED1;
        --pink-darker: #E790AB;
        --text-color: #333333;
    }
    
    /* Body and background */
    .stApp {
        background-color: var(--pink-light);
        color: var(--text-color);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #9E4B6C !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: var(--pink-medium-light);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--pink-dark);
        color: #4D1F33;
        border: none;
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--pink-darker);
        color: white;
        transform: scale(1.02);
    }
    
    /* Input fields - Ensure consistent styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div, .stMultiselect>div>div>div {
        border-color: var(--pink-medium) !important;
        background-color: white !important;
    }
    
    /* Set text color for selectbox elements */
    .stSelectbox div {
        color: var(--text-color) !important;
    }
    
    /* Input field labels */
    .stTextInput label, 
    .stNumberInput label, 
    .stSelectbox label, 
    .stMultiselect label, 
    .stSlider label {
        color: #9E4B6C !important;
        font-weight: 600;
        font-size: 15px;
    }
    
    /* Sliders */
    .stSlider>div>div>div {
        background-color: var(--pink-dark);
    }
    
    /* Hover tooltips */
    .stTooltipIcon {
        color: var(--pink-darker) !important;
    }
    
    /* Dataframe */
    .dataframe {
        border: 2px solid var(--pink-medium);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--pink-medium-light);
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color);
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--pink-dark) !important;
        color: #4D1F33 !important;
        font-weight: bold;
    }
    
    /* Cards or containers */
    .css-keje6w {
        background-color: white;
        border: 1px solid var(--pink-medium);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Alert messages */
    .stAlert {
        border-color: var(--pink-darker);
        background-color: var(--pink-medium);
    }
    
    /* Footer */
    footer {
        border-top: 1px solid var(--pink-medium);
        padding-top: 10px;
        text-align: center;
        color: gray;
    }
    
    /* Custom containers for major sections */
    .prediction-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid var(--pink-darker);
    }
    
    .visualization-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #9E4B6C;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: var(--pink-darker);
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Function to create a beautiful page header with ribbon design
def create_page_header():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px; margin-top: 20px;">
        <div style="display: inline-block; position: relative;">
            <h1 style="color: #9E4B6C; font-family: 'Helvetica Neue', sans-serif; font-size: 42px; 
                    padding: 20px 40px; background-color: #F2BED1; border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 0;">
                Breast Cancer Survival Prediction
            </h1>
            <div style="position: absolute; top: -15px; left: 50%; transform: translateX(-50%); 
                    background-color: #F9F5F6; padding: 5px 15px; border-radius: 20px; 
                    border: 2px solid #F2BED1;">
                <span style="font-weight: bold; color: #9E4B6C;">🎀 Hope & Healthcare 🎀</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Upload Breast_Cancer.csv", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not df.empty:
                return df
            else:
                st.error("The uploaded file is empty.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            st.stop()
    else:
        st.info("Please upload the Breast_Cancer.csv file to proceed.")
        st.stop()

# Function to preprocess data
def preprocess_data(df):
    # Create feature mappings
    stage_mapping = {'IIA': 1, 'IIB': 2, 'IIIA': 3, 'IIIB': 4, 'IIIC': 5}
    differentiate_mapping = {'Well differentiated': 3, 'Moderately differentiated': 2, 
                            'Poorly differentiated': 1, 'Undifferentiated': 0}
    status_mapping = {'Positive': 1, 'Negative': 0}
    stage_a_mapping = {'Regional': 1, 'Distant': 0}
    
    # Apply mappings
    processed_df = df.copy()
    
    # Apply mappings based on column existence
    if '6th Stage' in processed_df.columns:
        processed_df['6th Stage'] = processed_df['6th Stage'].map(stage_mapping)
    
    if 'Grade' in processed_df.columns:
        processed_df['Grade'] = processed_df['Grade'].map(differentiate_mapping)
    
    if 'A Stage' in processed_df.columns:
        processed_df['A Stage'] = processed_df['A Stage'].map(stage_a_mapping)
    
    if 'Estrogen Status' in processed_df.columns:
        processed_df['Estrogen Status'] = processed_df['Estrogen Status'].map(status_mapping)
    
    if 'Progesterone Status' in processed_df.columns:
        processed_df['Progesterone Status'] = processed_df['Progesterone Status'].map(status_mapping)
    
    # Make sure all expected columns exist
    expected_features = [
        'Age', '6th Stage', 'Grade', 'A Stage', 'Tumor Size', 
        'Estrogen Status', 'Progesterone Status', 'Regional Node Examined', 
        'Reginol Node Positive', 'Race_Black', 'Race_Other', 'Race_White', 
        'Marital Status_Divorced', 'Marital Status_Married', 'Marital Status_Separated', 
        'Marital Status_Single', 'Marital Status_Widowed'
    ]

    # Create any missing binary columns with default value 0
    for feature in expected_features:
        if feature not in processed_df.columns:
            if feature in ['Race_Black', 'Race_Other', 'Race_White', 
                        'Marital Status_Divorced', 'Marital Status_Married', 
                        'Marital Status_Separated', 'Marital Status_Single', 
                        'Marital Status_Widowed']:
                processed_df[feature] = 0
    
    # Handle target variable
    if 'Status' in processed_df.columns:
        processed_df['Status'] = processed_df['Status'].map({'Alive': 1, 'Dead': 0})
    
    return processed_df

# Function to train model
@st.cache_resource
def train_model(df):
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Create one-hot encoded columns for Race and Marital Status if they don't exist
    if 'Race_Black' not in processed_df.columns and 'Race' in processed_df.columns:
        race_dummies = pd.get_dummies(processed_df['Race'], prefix='Race')
        for col in race_dummies.columns:
            processed_df[col] = race_dummies[col]
            
    if 'Marital Status_Divorced' not in processed_df.columns and 'Marital Status' in processed_df.columns:
        marital_dummies = pd.get_dummies(processed_df['Marital Status'], prefix='Marital Status')
        for col in marital_dummies.columns:
            processed_df[col] = marital_dummies[col]
    
    # Select features and target
    features = [
        'Age', '6th Stage', 'Grade', 'A Stage', 'Tumor Size', 
        'Estrogen Status', 'Progesterone Status', 'Regional Node Examined', 
        'Reginol Node Positive', 'Race_Black', 'Race_Other', 'Race_White', 
        'Marital Status_Divorced', 'Marital Status_Married', 'Marital Status_Separated', 
        'Marital Status_Single', 'Marital Status_Widowed'
    ]
    
    # Use only available features
    available_features = [f for f in features if f in processed_df.columns]
    
    X = processed_df[available_features].fillna(0)  # Fill NAs with 0 for demonstration
    y = processed_df['Status'] if 'Status' in processed_df.columns else np.random.randint(0, 2, len(processed_df))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    alpha = 0.0001  # Default value
    model_nn = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', 
                          alpha=alpha, max_iter=200, random_state=42)
    model_nn.fit(X_train_scaled, y_train)
    
    return model_nn, scaler, available_features

# Create a function to generate a download link for CSV data
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none;color:#9E4B6C;font-weight:bold;">{text}</a>'
    return href

# Function to make a prediction
def make_prediction(model, scaler, features, input_data):
    # Process input data to match model features
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_proba = model.predict_proba(input_scaled)[0]
    prediction = model.predict(input_scaled)[0]
    
    return prediction, prediction_proba

# Function to create styled metric
def styled_metric(label, value, help_text=""):
    st.markdown(f"""
    <div style="background-color: white; border-radius: 10px; padding: 15px; 
             box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 10px; border-left: 3px solid #F2BED1;">
        <p style="color: #9E4B6C; font-size: 16px; margin-bottom: 5px; font-weight: bold;">{label}</p>
        <h3 style="font-size: 24px; margin: 0; color: #333;">{value}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if help_text:
        st.caption(help_text)

# Main function
def main():
    # Page header
    create_page_header()
    
    # Load data
    data = load_data()
    
    # Train model
    model, scaler, features = train_model(data)
    
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://i.pinimg.com/736x/2f/84/9b/2f849b0bbbd0ebb21697f44cd7f72c75.jpg", width=250)
        page = st.radio("", ["Home", "Prediction", "Data Visualization", "About"])
        
        st.markdown("---")
        
        # Sidebar info
        st.markdown("### 🎀 Pink Ribbon Campaign")
        st.markdown("""
        Early detection is key to successful treatment. Regular check-ups save lives.
        
        Learn more about breast cancer awareness and prevention.
        """)
        
        # Add sidebar instructions
        st.markdown("---")
        st.markdown("### How to use this app:")
        st.markdown("""
        1. Use the **Prediction** page to input patient data
        2. Explore the **Data Visualization** page to see patterns
        3. Learn more on the **About** page
        """)
    
    # Home page
    if page == "Home":
        # Two-column layout for welcome message
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## Welcome to the Breast Cancer Survival Prediction Tool")
            st.markdown("""
            This application uses machine learning to predict breast cancer patient survival based on clinical features.
            Our neural network model analyzes patient characteristics to estimate survival probability.
            
            ### Key Features:
            
            * **Predict survival chances** based on patient data
            * **Explore interactive visualizations** showing relationships between clinical features
            * **Learn about relationships** between tumor characteristics and survival outcomes
            
            Navigate using the sidebar to access different sections of the application.
            """)
            
            # Call-to-action buttons
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("Start Prediction", key="home_predict"):
                    st.session_state.page = "Prediction"
                    st.rerun()
            with col1b:
                if st.button("Explore Data", key="home_explore"):
                    st.session_state.page = "Data Visualization"
                    st.rerun()
            
        with col2:
            # Simple statistics
            st.markdown("### Dataset Summary")
            
            total_patients = len(data)
            survivors = len(data[data['Status'] == 'Alive']) if 'Status' in data.columns else 0
            survival_rate = f"{survivors / total_patients:.1%}" if total_patients > 0 else "N/A"
            
            styled_metric("Total Patients", f"{total_patients:,}")
            styled_metric("Survivors", f"{survivors:,}")
            styled_metric("Overall Survival Rate", survival_rate)
        
        # Overview stats in cards
        st.markdown("## Quick Insights")
        
        # Create three columns for stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_age = data['Age'].mean() if 'Age' in data.columns else 0
            st.markdown("""
            <div style="background-color: #F8E8EE; border-radius: 10px; padding: 20px; text-align: center;">
                <h3 style="margin-bottom: 10px; color: #9E4B6C;">Average Age</h3>
                <p style="font-size: 28px; font-weight: bold; margin: 0; color: #333333;">{:.1f}</p>
                <p style="color: #666666; margin-top: 5px;">years</p>
            </div>
            """.format(avg_age), unsafe_allow_html=True)
        
        with col2:
            avg_tumor = data['Tumor Size'].mean() if 'Tumor Size' in data.columns else 0
            st.markdown("""
            <div style="background-color: #FDCEDF; border-radius: 10px; padding: 20px; text-align: center;">
                <h3 style="margin-bottom: 10px; color: #9E4B6C;">Average Tumor Size</h3>
                <p style="font-size: 28px; font-weight: bold; margin: 0; color: #333333;">{:.1f}</p>
                <p style="color: #666666; margin-top: 5px;">mm</p>
            </div>
            """.format(avg_tumor), unsafe_allow_html=True)
        
        with col3:
            avg_survival = data['Survival Months'].mean() if 'Survival Months' in data.columns else 0
            st.markdown("""
            <div style="background-color: #F2BED1; border-radius: 10px; padding: 20px; text-align: center;">
                <h3 style="margin-bottom: 10px; color: #9E4B6C;">Average Survival</h3>
                <p style="font-size: 28px; font-weight: bold; margin: 0; color: #333333;">{:.1f}</p>
                <p style="color: #666666; margin-top: 5px;">months</p>
            </div>
            """.format(avg_survival), unsafe_allow_html=True)
        
        # Sample visualization
        st.markdown("## Featured Visualization")
        
        # Create scatter plot of tumor size vs survival months
        if 'Tumor Size' in data.columns and 'Survival Months' in data.columns and 'Status' in data.columns:
            scatter_data = data.copy()
            scatter_data['Tumor Size Safe'] = scatter_data['Tumor Size'].clip(lower=0.1)
            fig = px.scatter(
                data,
                x='Tumor Size',
                y='Survival Months',
                color='Status',
                size='Tumor Size',
                size_max=18,
                color_discrete_sequence=['#E790AB', '#9E4B6C'],
                hover_data={'Tumor Size': ':.2f', 'Survival Months': ':.1f'},
                title='Tumor Size vs Survival Months by Status',
                labels={
                    'Tumor Size': 'Tumor Size (mm)',
                    'Survival Months': 'Survival Months',
                    'Status': 'Status'
                }
            )
            
            fig.update_traces(
                marker=dict(opacity=0.7, line=dict(width=1, color='black')),
                selector=dict(mode='markers')
            )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                legend_title_text='Status',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=13),
                xaxis=dict(
                    title='Tumor Size (mm)',
                    gridcolor='lightgray',
                    zeroline=False
                ),
                yaxis=dict(
                    title='Survival Months',
                    gridcolor='lightgray',
                    zeroline=False
                ),
                title=dict(
                    font_size=18,
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sample data doesn't contain all required columns for this visualization.")
    
    # Prediction page
    elif page == "Prediction":
        st.markdown("## Breast Cancer Survival Prediction")
        st.markdown("Enter patient information to predict survival probability.")
        
        # Make the form look nicer with a background
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; 
                  box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
                  border-left: 5px solid #F2BED1;">
        <h3 style="color: #9E4B6C; margin-top: 0;">Patient Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a form for input
        with st.form("prediction_form"):
            # Organize inputs into columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=20, max_value=100, value=55, help="Patient's age in years")
                tumor_size = st.number_input("Tumor Size (mm)", min_value=1.0, max_value=150.0, value=25.0, step=0.1, help="Size of primary tumor in millimeters")
                stage_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"], index=0, help="Cancer stage according to the 6th edition staging system")
                grade = st.selectbox("Grade", ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"], index=1, help="Degree of cancer cell differentiation")
                
            with col2:
                estrogen_status = st.selectbox("Estrogen Status", ["Positive", "Negative"], index=0, help="Estrogen receptor status")
                progesterone_status = st.selectbox("Progesterone Status", ["Positive", "Negative"], index=0, help="Progesterone receptor status")
                a_stage = st.selectbox("A Stage", ["Regional", "Distant"], index=0, help="Stage classification as regional or distant")
                regional_nodes = st.number_input("Regional Nodes Examined", min_value=0, max_value=50, value=5, help="Number of regional lymph nodes examined")
            
            with col3:
                positive_nodes = st.number_input("Regional Nodes Positive", min_value=0, max_value=50, value=1, help="Number of regional lymph nodes positive for tumor cells")
                race = st.selectbox("Race", ["White", "Black", "Other"], index=0, help="Patient's racial background")
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Separated", "Widowed"], index=1, help="Patient's marital status")
            
            # Create a styled submit button
            submitted = st.form_submit_button("Predict Survival")
        
        # Process the form submission
        if submitted:
            # Prepare input data
            # Map categorical variables
            stage_mapping = {'IIA': 1, 'IIB': 2, 'IIIA': 3, 'IIIB': 4, 'IIIC': 5}
            differentiate_mapping = {'Well differentiated': 3, 'Moderately differentiated': 2, 
                                 'Poorly differentiated': 1, 'Undifferentiated': 0}
            status_mapping = {'Positive': 1, 'Negative': 0}
            stage_a_mapping = {'Regional': 1, 'Distant': 0}
            
            # Create one-hot encoded race
            race_black = 1 if race == "Black" else 0
            race_other = 1 if race == "Other" else 0
            race_white = 1 if race == "White" else 0
            
            # Create one-hot encoded marital status
            marital_divorced = 1 if marital_status == "Divorced" else 0
            marital_married = 1 if marital_status == "Married" else 0
            marital_separated = 1 if marital_status == "Separated" else 0
            marital_single = 1 if marital_status == "Single" else 0
            marital_widowed = 1 if marital_status == "Widowed" else 0
            
            # Create input dictionary
            input_data = {
                'Age': age,
                '6th Stage': stage_mapping[stage_6th],
                'Grade': differentiate_mapping[grade],
                'A Stage': stage_a_mapping[a_stage],
                'Tumor Size': tumor_size,
                'Estrogen Status': status_mapping[estrogen_status],
                'Progesterone Status': status_mapping[progesterone_status],
                'Regional Node Examined': regional_nodes,
                'Reginol Node Positive': positive_nodes,
                'Race_Black': race_black,
                'Race_Other': race_other,
                'Race_White': race_white,
                'Marital Status_Divorced': marital_divorced,
                'Marital Status_Married': marital_married,
                'Marital Status_Separated': marital_separated,
                'Marital Status_Single': marital_single,
                'Marital Status_Widowed': marital_widowed
            }
            
            # Make prediction
            prediction, prediction_proba = make_prediction(model, scaler, features, input_data)
            
            # Display prediction result
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 10px; 
                      box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 30px; margin-bottom: 20px;
                      border-left: 5px solid #E790AB;">
            <h3 style="color: #9E4B6C; margin-top: 0;">Prediction Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for results display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display prediction outcome
                survival_status = "Likely to Survive" if prediction == 1 else "May Not Survive"
                
                # Circle indicator with prediction
                survival_color = "#28a745" if prediction == 1 else "#dc3545"
                survival_proba = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: white; 
                          border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="position: relative; width: 200px; height: 200px; margin: 0 auto;">
                        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                                  border-radius: 50%; background-color: {survival_color}; opacity: 0.2;"></div>
                        <div style="position: absolute; top: 15px; left: 15px; width: calc(100% - 30px); height: calc(100% - 30px);
                                  border-radius: 50%; background-color: white; display: flex; flex-direction: column;
                                  justify-content: center; align-items: center; text-align: center;">
                            <p style="margin: 0; font-size: 18px; color: #666;">Prediction</p>
                            <h2 style="margin: 5px 0; color: {survival_color};">{survival_status}</h2>
                            <p style="margin: 0; font-size: 24px; font-weight: bold; color: {survival_color};">{survival_proba:.1%}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Additional information
                st.markdown("<h4 style='color: #9E4B6C;'>Key Factors Influencing Prediction</h4>", unsafe_allow_html=True)
                
                # Show importance of different factors
                st.markdown("<b>Age:</b> Higher age can increase risk", unsafe_allow_html=True)
                st.progress(min(age/100, 1.0))
                
                st.markdown("<b>Tumor Size:</b> Larger tumors may indicate higher risk", unsafe_allow_html=True)
                st.progress(min(tumor_size/50, 1.0))
                
                st.markdown("<b>Cancer Stage:</b> Advanced stages have higher risk", unsafe_allow_html=True)
                stage_progress = stage_mapping[stage_6th] / 5
                st.progress(stage_progress)
                
                st.markdown("<b>Hormone Receptor Status:</b> Negative receptors may indicate higher risk", unsafe_allow_html=True)
                hormone_progress = 0.3 if estrogen_status == "Positive" else 0.8
                st.progress(hormone_progress)
                
                # Recommendations
                st.markdown("""
                <div style="background-color: #F8E8EE; padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <h4 style="color: #9E4B6C; margin-top: 0;">Recommendations</h4>
                    <ul style="margin-bottom: 0;">
                        <li>Consult with your oncologist about these results</li>
                        <li>Consider follow-up tests as recommended by your healthcare provider</li>
                        <li>Regular monitoring and follow-up appointments are essential</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Data Visualization page
    elif page == "Data Visualization":
        st.markdown("## Data Visualization Dashboard")
        st.markdown("Explore patterns and relationships in breast cancer data.")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Tumor vs Survival", "Age Analysis", "Distribution Analysis", "Stage Analysis"])
        
        with tabs[0]:
            st.markdown("### Tumor Size vs. Survival Months")
            
            # Visualization 1: Scatter plot of tumor size vs survival months
            if 'Tumor Size' in data.columns and 'Survival Months' in data.columns and 'Status' in data.columns:
                fig = px.scatter(
                    data,
                    x='Tumor Size',
                    y='Survival Months',
                    color='Status',
                    symbol='Status',
                    size='Tumor Size',
                    size_max=18,
                    color_discrete_sequence=['#E790AB', '#9E4B6C'],
                    hover_data={
                        '6th Stage': True,
                        'Grade': True,
                        'Age': True,
                        'Race': True,
                        'Tumor Size': ':.2f',
                        'Survival Months': ':.1f',
                        'Status': False
                    },
                    title='Tumor Size vs Survival Months by Survival Status',
                    labels={
                        'Tumor Size': 'Tumor Size (mm)',
                        'Survival Months': 'Survival Months',
                        'Status': 'Survival Status'
                    }
                )

                fig.update_traces(
                    marker=dict(opacity=0.7, line=dict(width=1, color='black')),
                    selector=dict(mode='markers')
                )

                fig.update_layout(
                    height=600,
                    showlegend=True,
                    legend_title_text='Survival Status',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=13),
                    xaxis=dict(
                        title='Tumor Size (mm)',
                        gridcolor='lightgray',
                        zeroline=False
                    ),
                    yaxis=dict(
                        title='Survival Months',
                        gridcolor='lightgray',
                        zeroline=False
                    ),
                    title=dict(
                        font_size=18,
                        x=0.5
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                <div style="background-color: #F9F5F6; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <h4 style="color: #9E4B6C; margin-top: 0;">Insights</h4>
                    <p>This scatter plot shows the relationship between tumor size and survival months, with points colored by survival status. 
                    Larger tumors often correlate with shorter survival times, though there are exceptions to this pattern.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sample data doesn't contain all required columns for this visualization.")
        
        with tabs[1]:
            st.markdown("### Age Analysis")
            
            # Visualization 2: Box plot for age by survival status
            if 'Age' in data.columns and 'Status' in data.columns:
                fig = px.box(
                    data,
                    x='Status',
                    y='Age',
                    color='Status',
                    color_discrete_sequence=['#E790AB', '#9E4B6C'],
                    title='Age Distribution by Survival Status',
                    labels={
                        'Age': 'Age (years)',
                        'Status': 'Survival Status'
                    }
                )

                fig.update_traces(
                    marker=dict(size=8, opacity=0.8, line=dict(width=1, color='black')),
                    line=dict(width=2)
                )

                fig.update_layout(
                    height=500,
                    showlegend=True,
                    legend=dict(
                        title='Survival Status',
                        bordercolor='black',
                        borderwidth=1,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        font=dict(size=12)
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=14),
                    title=dict(
                        text='Age Distribution by Survival Status',
                        font=dict(size=20, family='Arial', color='black'),
                        x=0.5
                    ),
                    xaxis=dict(
                        title='Survival Status',
                        title_font=dict(size=16, family='Arial', color='black'),
                        tickfont=dict(size=14, family='Arial'),
                        gridcolor='lightgray',
                        zeroline=False
                    ),
                    yaxis=dict(
                        title='Age (years)',
                        title_font=dict(size=16, family='Arial', color='black'),
                        tickfont=dict(size=14, family='Arial'),
                        gridcolor='lightgray',
                        zeroline=False
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                <div style="background-color: #F9F5F6; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <h4 style="color: #9E4B6C; margin-top: 0;">Insights</h4>
                    <p>This box plot shows the age distribution for patients who survived versus those who did not. 
                    The analysis helps identify whether age is a significant factor in survival outcomes for breast cancer patients.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sample data doesn't contain all required columns for this visualization.")
        
        with tabs[2]:
            st.markdown("### Distribution Analysis")
            
            # Visualization 3: Histograms with KDE for age and tumor size
            if 'Age' in data.columns and 'Tumor Size' in data.columns:
                # Create subplots
                fig = make_subplots(
                    rows=1, 
                    cols=2, 
                    subplot_titles=['Age Distribution', 'Tumor Size Distribution'],
                    horizontal_spacing=0.2
                )

                # Age histogram
                age_hist = go.Histogram(
                    x=data['Age'],
                    nbinsx=10,
                    histnorm='probability density',
                    name='Age Histogram',
                    marker=dict(color='#FDCEDF', line=dict(color='#F2BED1', width=1)),
                    opacity=0.7
                )
                fig.add_trace(age_hist, row=1, col=1)

                # Age KDE (approximation)
                age_steps = np.linspace(data['Age'].min(), data['Age'].max(), 100)
                kde_age = np.exp(-(age_steps - data['Age'].mean())**2 / (2 * data['Age'].std()**2)) / (data['Age'].std() * np.sqrt(2 * np.pi))
                
                age_kde = go.Scatter(
                    x=age_steps,
                    y=kde_age,
                    mode='lines',
                    name='Age KDE',
                    line=dict(color='#9E4B6C', width=3)
                )
                fig.add_trace(age_kde, row=1, col=1)

                # Tumor size histogram
                tumor_hist = go.Histogram(
                    x=data['Tumor Size'],
                    nbinsx=10,
                    histnorm='probability density',
                    name='Tumor Size Histogram',
                    marker=dict(color='#F8E8EE', line=dict(color='#E790AB', width=1)),
                    opacity=0.7
                )
                fig.add_trace(tumor_hist, row=1, col=2)

                # Tumor size KDE (approximation)
                tumor_steps = np.linspace(data['Tumor Size'].min(), data['Tumor Size'].max(), 100)
                kde_tumor = np.exp(-(tumor_steps - data['Tumor Size'].mean())**2 / (2 * data['Tumor Size'].std()**2)) / (data['Tumor Size'].std() * np.sqrt(2 * np.pi))
                
                tumor_kde = go.Scatter(
                    x=tumor_steps,
                    y=kde_tumor,
                    mode='lines',
                    name='Tumor Size KDE',
                    line=dict(color='#9E4B6C', width=3)
                )
                fig.add_trace(tumor_kde, row=1, col=2)

                # Update layout
                fig.update_layout(
                    title=dict(
                        text='Distributions of Age and Tumor Size',
                        font=dict(size=20, family='Arial', color='#9E4B6C'),
                        x=0.5
                    ),
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        x=0.5,
                        y=-0.2,
                        xanchor='center',
                        font=dict(size=12),
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='#F2BED1',
                        borderwidth=1
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=14),
                    bargap=0.3
                )

                # Update axes
                fig.update_xaxes(
                    title_text='Age (years)',
                    row=1, col=1,
                    gridcolor='lightgray',
                    zeroline=False,
                    title_font=dict(size=16, color='#9E4B6C'),
                    tickfont=dict(size=12)
                )
                fig.update_xaxes(
                    title_text='Tumor Size (mm)',
                    row=1, col=2,
                    gridcolor='lightgray',
                    zeroline=False,
                    title_font=dict(size=16, color='#9E4B6C'),
                    tickfont=dict(size=12)
                )

                fig.update_yaxes(
                    title_text='Probability Density',
                    row=1, col=1,
                    gridcolor='lightgray',
                    zeroline=False,
                    title_font=dict(size=16, color='#9E4B6C'),
                    tickfont=dict(size=12)
                )
                fig.update_yaxes(
                    title_text='Probability Density',
                    row=1, col=2,
                    gridcolor='lightgray',
                    zeroline=False,
                    title_font=dict(size=16, color='#9E4B6C'),
                    tickfont=dict(size=12)
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                <div style="background-color: #F9F5F6; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <h4 style="color: #9E4B6C; margin-top: 0;">Insights</h4>
                    <p>These histograms with kernel density estimation (KDE) curves show the distribution of patient ages and tumor sizes in the dataset.
                    Understanding these distributions helps identify the most common age groups affected and typical tumor sizes at diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sample data doesn't contain all required columns for this visualization.")
                
        with tabs[3]:
            st.markdown("### Stage Analysis")
            
            # Visualization 4: Survival months by T Stage and Estrogen Status
            if '6th Stage' in data.columns and 'Estrogen Status' in data.columns and 'Survival Months' in data.columns:
                # Group by T Stage and Estrogen Status
                grouped_data = data.groupby(['6th Stage', 'Estrogen Status'])['Survival Months'].mean().reset_index()
                
                # Create the visualization
                fig = px.bar(
                    grouped_data,
                    x='6th Stage',
                    y='Survival Months',
                    color='Estrogen Status',
                    barmode='group',
                    color_discrete_sequence=['#E790AB', '#9E4B6C'],
                    title='Average Survival Months by Stage and Estrogen Status',
                    labels={
                        '6th Stage': 'Cancer Stage',
                        'Survival Months': 'Average Survival (Months)',
                        'Estrogen Status': 'Estrogen Status'
                    }
                )
                
                fig.update_layout(
                    height=500,
                    showlegend=True,
                    legend=dict(
                        title='Estrogen Status',
                        bordercolor='black',
                        borderwidth=1,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        font=dict(size=12)
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=14),
                    title=dict(
                        text='Average Survival Months by Stage and Estrogen Status',
                        font=dict(size=20, family='Arial', color='#9E4B6C'),
                        x=0.5
                    ),
                    xaxis=dict(
                        title='Cancer Stage',
                        title_font=dict(size=16, family='Arial', color='#9E4B6C'),
                        tickfont=dict(size=14, family='Arial'),
                        gridcolor='lightgray',
                        zeroline=False
                    ),
                    yaxis=dict(
                        title='Average Survival (Months)',
                        title_font=dict(size=16, family='Arial', color='#9E4B6C'),
                        tickfont=dict(size=14, family='Arial'),
                        gridcolor='lightgray',
                        zeroline=False
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                <div style="background-color: #F9F5F6; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <h4 style="color: #9E4B6C; margin-top: 0;">Insights</h4>
                    <p>This visualization shows how survival months vary across different cancer stages and estrogen status.
                    It helps identify which combinations of factors may be associated with better or worse survival outcomes.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sample data doesn't contain all required columns for this visualization.")
                
        # Data preview section
        st.markdown("### Data Preview")
        if st.checkbox("Show raw data", value=False):
            st.dataframe(data.head(10), use_container_width=True)
            
            # Add download button for data
            st.markdown(get_table_download_link(data, "breast_cancer_data.csv", "📥 Download full dataset"), unsafe_allow_html=True)
    
    # About page
    elif page == "About":
        st.markdown("## About This Application")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Breast Cancer Survival Prediction Tool

            This tool helps estimate breast cancer survival chances using machine learning. By analyzing clinical information, it offers insights into factors that may affect patient outcomes.

            ### How It Works

            We use a Neural Network model with:
            - One hidden layer (10 neurons)
            - Logistic activation
            - Regularization (alpha = 0.0001)
            - 200 training iterations

            ### What the Model Considers

            - **Patient Info**: Age, race, marital status  
            - **Tumor Details**: Size, grade, stage  
            - **Hormone Receptors**: Estrogen & progesterone status  
            - **Lymph Nodes**: Examined and positive count

            """) 
        with col2:
            # Display ribbon image
            st.image("https://i.pinimg.com/736x/15/9b/38/159b38ddda80e203afe1d11639dc96c0.jpg", width=250)
            
        # Team information
        st.markdown("### Development Team")
        
        # Use columns for team members
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **ENG. Sherry Rafiq**
            """)
            
        with col2:
            st.markdown(""" 
            **ENG. Rania Elsayed**
            """)
            
        with col3:
            st.markdown(""" 
            **ENG. Salsabil Waleed** 
            """)
    
    # Footer
    st.markdown("""
    <footer style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #F2BED1;">
        <p>💕 Breast Cancer Survival Prediction Tool • Developed with ❤️ for cancer research</p>
        <p style="font-size: 12px; color: #666;">© 2025 • Version 1.0</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()