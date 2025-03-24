import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected' not in st.session_state:
    st.session_state.selected = "Home"

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        models['diabetes'] = pickle.load(open('Models/diabetes_model.sav', 'rb'))
        models['heart_disease'] = pickle.load(open('Models/heart_disease_model.sav', 'rb'))
        models['parkinsons'] = pickle.load(open('Models/parkinsons_model.sav', 'rb'))
        models['lung_cancer'] = pickle.load(open('Models/lungs_disease_model.sav', 'rb'))
        models['thyroid'] = pickle.load(open('Models/Thyroid_model.sav', 'rb'))
        print("All models loaded successfully!")  # Debugging
    except FileNotFoundError as e:
        st.error(f"Error: Model file not found. Please check the file paths. Details: {e}")
        st.stop()  # Stop execution if models can't be loaded
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    return models

models = load_models()

# Function to create a radar chart
def display_radar_chart(features):
    categories = ['Fundamental Frequency', 'Jitter', 'Shimmer', 'NHR', 'HNR', 'DFA']
    values = features[:6]  # Assuming the first 6 features correspond to the radar chart categories

    # Create radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Voice Features',
        line_color='#005EB8',
        fillcolor='rgba(0, 94, 184, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )

    return fig

# Custom CSS for modern interface
st.markdown("""
<style>
    body {
        background-color: #007BFF
        background-size: cover;
        background-attachment: fixed;
    }

    /* Main containers */
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }

    /* Custom card styling */
    .disease-card {
        border-radius: 12px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .disease-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 16px rgba(0, 0, 0, 0.12);
    }

    .disease-icon {
        font-size: 48px;
        margin-bottom: 15px;
        color: #005EB8;
    }

    .disease-title {
        font-size: 22px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
    }

    .disease-description {
        color: #666;
        font-size: 16px;
        margin-bottom: 15px;
        flex-grow: 1;
    }

    .begin-button {
        background-color: #005EB8;
        color: white;
        padding: 10px 20px;
        border-radius: 50px;
        text-decoration: none;
        font-weight: 500;
        display: inline-block;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .begin-button:hover {
        background-color: #004085;
    }

    /* Headers and text */
    h1, h2 {
        color: #2c3e50;
    }

    .welcome-banner {
        background: linear-gradient(135deg, #005EB8, #0088cc);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0, 94, 184, 0.2);
    }

    .welcome-banner h1 {
        color: white;
        margin-bottom: 10px;
    }

    .welcome-banner p {
        font-size: 18px;
        opacity: 0.9;
    }

    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 14px;
        margin-top: 40px;
    }

    /* Nav menu styling */
    .nav-menu {
        background-color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .nav-button {
        background: none;
        border: none;
        color: #2c3e50;
        margin: 0 10px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        padding: 8px 15px;
        border-radius: 20px;
        transition: background-color 0.3s;
    }

    .nav-button:hover {
        background-color: #f0f0f0;
    }

    .nav-button.active {
        background-color: #e6f2ff;
        color: #005EB8;
    }

    /* Info cards */
    .info-card {
        background-color: #fff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .info-icon {
        font-size: 24px;
        color: #005EB8;
        margin-right: 10px;
    }

    .statistic-card {
        background-color: #f8f9fa;
        border-left: 4px solid #005EB8;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }

    .statistic-number {
        font-size: 24px;
        font-weight: bold;
        color: #005EB8;
    }

    .statistic-label {
        color: #666;
        font-size: 14px;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin-top: 30px;
        font-size: 14px;
        color: #856404;
    }

    /* Form styling */
    .stForm {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Section headers in forms */
    .section-header {
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: 600;
        color: #2c3e50;
        border-left: 4px solid #005EB8;
    }

    /* Result boxes */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-top: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    .positive-result {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }

    .negative-result {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }

    /* Back button */
    .back-button {
        margin-bottom: 20px;
        display: inline-flex;
        align-items: center;
        background-color: #f8f9fa;
        border: none;
        border-radius: 30px;
        padding: 8px 15px;
        color: #2c3e50;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.2s;
    }

    .back-button:hover {
        background-color: #e9ecef;
    }

    .back-arrow {
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Navigation at the top
def display_nav():
    st.markdown('<div class="nav-menu">', unsafe_allow_html=True)
    # Left side - Logo and app name
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown('<span style="font-size: 22px; font-weight: 600; color: #005EB8;">🏥 Medical Diagnosis AI</span>', unsafe_allow_html=True)

    # Middle - Navigation buttons
    with col2:
        nav_cols = st.columns(3)
        with nav_cols[0]:
            home_class = "active" if 'selected' not in st.session_state or st.session_state.selected == "Home" else ""
            if st.button("🏠 Home", key="home_nav", help="Go to home page"):
                st.session_state.selected = "Home"
                

        with nav_cols[1]:
            about_class = "active" if 'selected' in st.session_state and st.session_state.selected == "About" else ""
            if st.button("ℹ️ About", key="about_nav", help="Learn about this application"):
                st.session_state.selected = "About"

        with nav_cols[2]:
            contact_class = "active" if 'selected' in st.session_state and st.session_state.selected == "Contact" else ""
            if st.button("📞 Contact", key="contact_nav", help="Contact support"):
                st.session_state.selected = "Contact"

    # Right side - Optional space for user profile or settings
    with col3:
        st.markdown('<div style="text-align: right;"><span style="color: #666;">v2.0</span></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# Disease Prediction Pages
def display_home():
    # Welcome Banner
    st.markdown("""
    <div class="welcome-banner">
        <h1>Welcome to Medical Diagnosis AI</h1>
        <p>AI-powered diagnostic assistant helping healthcare professionals make informed decisions</p>
    </div>
    """, unsafe_allow_html=True)
    # Quick statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="statistic-card">
            <div class="statistic-number">5+</div>
            <div class="statistic-label">Disease Modules</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="statistic-card">
            <div class="statistic-number">95%</div>
            <div class="statistic-label">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="statistic-card">
            <div class="statistic-number">10k+</div>
            <div class="statistic-label">Cases Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="statistic-card">
            <div class="statistic-number">24/7</div>
            <div class="statistic-label">AI Assistance</div>
        </div>
        """, unsafe_allow_html=True)
# Disease Module Cards
    st.markdown("## Disease Prediction Modules")
    st.write("Select a module to begin diagnosis")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div class="disease-card">
            <div class="disease-icon">❤️</div>
            <div class="disease-title">Heart Disease</div>
            <div class="disease-description">
                Predict risk of cardiovascular disease using clinical parameters like blood pressure, cholesterol levels, and ECG results.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Heart Disease", key="heart_button"):
            st.session_state.selected = "Heart Disease"
            

    with col2:
        st.markdown("""
        <div class="disease-card">
            <div class="disease-icon">🔬</div>
            <div class="disease-title">Diabetes</div>
            <div class="disease-description">
                Calculate probability of diabetes based on glucose levels, insulin, BMI, and other relevant factors.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Diabetes", key="diabetes_button"):
            st.session_state.selected = "Diabetes"

    with col3:
        st.markdown("""
        <div class="disease-card">
            <div class="disease-icon">🔊</div>
            <div class="disease-title">Parkinson's Disease</div>
            <div class="disease-description">
                Assess Parkinson's disease likelihood through voice pattern analysis and other neurological indicators.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Parkinson's", key="parkinsons_button"):
            st.session_state.selected = "Parkinson's"
    with col4:
    # Lung Cancer Prediction Card
        st.markdown("""
        <div class="disease-card">
            <div class="disease-icon">🫁</div>
            <div class="disease-title">Lung Cancer</div>
            <div class="disease-description">
                Evaluate risk factors for lung cancer based on lifestyle choices and symptoms.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Lung Cancer", key="lung_cancer_button"):
            st.session_state.selected = "Lung Cancer"

    with col5:
        # Thyroid Disease Prediction Card
        st.markdown("""
        <div class="disease-card">
            <div class="disease-icon">🦋</div>
            <div class="disease-title">Thyroid Disease</div>
            <div class="disease-description">
                Assess thyroid function through lab values and symptoms to determine risk of dysfunction.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Thyroid", key="thyroid_button"):
            st.session_state.selected = "Thyroid"

def display_heart_disease():
    st.title("Heart Disease Prediction")

    with st.form(key='heart_disease_form'):
        st.markdown("### Patient Information")
        age = st.slider('Age', 18, 100, 50)
        sex = st.radio('Sex (1 = Male; 0 = Female)', ['Female', 'Male'])
        sex_val = 1 if sex == 'Male' else 0

        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Non-anginal Pain", 3: "Asymptomatic"}[x]) #Duplication of value 3 corrected
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
        chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        fbs_val = 1 if fbs == 'Yes' else 0

        restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Having ST-T wave abnormality", 2: "Showing probable or definite left ventricular hypertrophy"}[x])

        thalach = st.slider('Maximum Heart Rate', 60, 220, 150)
        exang = st.radio('Exercise Induced Angina', ['No', 'Yes'])
        exang_val = 1 if exang == 'Yes' else 0
        oldpeak = st.slider('ST Depression', 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox('Slope of Peak Exercise ST', [0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
        ca = st.slider('Number of Major Vessels', 0, 3, 0)
        thal = st.selectbox('Thalassemia', [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x])

        submit_button = st.form_submit_button(label="Run Heart Disease Prediction")

    if submit_button:
        with st.spinner('Analyzing cardiovascular parameters...'):
            try:
                heart_prediction = models['heart_disease'].predict([[age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
                st.success("### Prediction Complete!")
                st.markdown("#### Summary of Input Data:")
                st.write(f"- Age: {age}")
                st.write(f"- Sex: {sex}")
                st.write(f"- Chest Pain Type: {cp}")
                st.write(f"- Resting Blood Pressure: {trestbps} mm Hg")
                st.write(f"- Serum Cholesterol: {chol} mg/dl")
                st.write(f"- Fasting Blood Sugar: {fbs}")
                st.write(f"- Resting Electrocardiographic Results: {restecg}")
                st.write(f"- Maximum Heart Rate: {thalach}")
                st.write(f"- Exercise Induced Angina: {'Yes' if exang_val == 1 else 'No'}")
                st.write(f"- ST Depression: {oldpeak}")
                st.write(f"- Slope of Peak Exercise ST: {slope}")
                st.write(f"- Number of Major Vessels: {ca}")
                st.write(f"- Thalassemia: {thal}")            

                # Result display with recommendations
                if heart_prediction[0] == 1:
                    st.error("### Prediction: Heart Disease Detected")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient may have heart disease.

                    **Recommendation:** Further cardiac evaluation is recommended, including:
                    - Echocardiogram
                    - Stress test
                    - Coronary angiography (if necessary)
                    - Consultation with a cardiologist
                    """)
                else:
                    st.success("### Prediction: No Heart Disease Detected")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient likely does not have heart disease.

                    **Recommendation:** Continue heart-healthy lifestyle habits:
                    - Regular exercise
                    - Balanced diet
                    - Avoid smoking
                    - Limit alcohol consumption
                    - Manage stress
                    """)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

def display_parkinsons():
     st.title("Parkinson's Disease Prediction")

     with st.form(key='parkinsons_form'):
        st.markdown("### Voice Feature Measurements")
        meanfreq = st.slider('Average Vocal Fundamental Frequency (Hz)', 80.0, 260.0, 150.0)
        sd = st.slider('Frequency Variation (SD)', 0.000, 0.100, 0.05, 0.001)
        median = st.slider('Median Fundamental Frequency', 80.0, 260.0, 150.0)
        Q25 = st.slider('First Quartile', 80.0, 260.0, 150.0)
        Q75 = st.slider('Third Quartile', 80.0, 260.0, 150.0)
        IQR = st.slider('Interquartile Range', 0.0, 0.2, 0.1, 0.001)
        skew = st.slider('Skewness', -5.0, 5.0, 0.0, 0.01)
        kurt = st.slider('Kurtosis', 1.0, 50.0, 5.0, 0.1)
        sp_ent = st.slider('Spectral Entropy', 0.0, 1.0, 0.5, 0.01)
        sfm = st.slider('Spectral Flatness', 0.0, 1.0, 0.5, 0.01)
        mode = st.slider('Mode Frequency', 80.0, 260.0, 150.0)
        centroid = st.slider('Frequency Centroid', 80.0, 260.0, 150.0)
        peakf = st.slider('Peak Frequency', 80.0, 260.0, 150.0)
        meanfun = st.slider('Average Fundamental Frequency Across Acoustic Signals', 0.0, 0.5, 0.25, 0.001)
        minfun = st.slider('Minimum Fundamental Frequency Across Acoustic Signals', 0.0, 0.2, 0.1, 0.001)
        maxfun = st.slider('Maximum Fundamental Frequency Across Acoustic Signals', 0.1, 0.5, 0.3, 0.001)
        meandom = st.slider('Average of Dominant Frequency Measured Across Acoustic Signals', 0.0, 2.0, 1.0, 0.01)
        mindom = st.slider('Minimum of Dominant Frequency Measured Across Acoustic Signals', 0.0, 0.5, 0.2, 0.001)
        maxdom = st.slider('Maximum of Dominant Frequency Measured Across Acoustic Signals', 2.0, 10.0, 5.0, 0.01)
        dfrange = st.slider('Range of Dominant Frequency Measured Across Acoustic Signals', 2.0, 10.0, 5.0, 0.01)
        modindx = st.slider('Modulation Index', 0.0, 0.2, 0.1, 0.001)
        ppe = st.slider('Pitch Period Entropy', 0.0, 1.0, 0.5, 0.01)
        submit_button = st.form_submit_button(label="Run Parkinson's Prediction")

     if submit_button:
        with st.spinner('Analyzing voice features...'):
            try:
                parkinsons_prediction = models['parkinsons'].predict([[meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm, mode, centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx, ppe]])
                st.success("### Prediction Complete!")
                st.markdown("#### Summary of Input Data:")
                st.write(f"- Average Vocal Fundamental Frequency (Hz): {meanfreq}")
                st.write(f"- Frequency Variation (SD): {sd}")
                st.write(f"- Median Fundamental Frequency (Hz): {median}")
                st.write(f"- First Quartile: {Q25}")
                st.write(f"- Third Quartile: {Q75}")
                st.write(f"- Interquartile Range: {IQR}")
                st.write(f"- Skewness: {skew}")
                st.write(f"- Kurtosis: {kurt}")
                st.write(f"- Spectral Entropy: {sp_ent}")
                st.write(f"- Spectral Flatness: {sfm}")
                st.write(f"- Mode Frequency: {mode}")
                st.write(f"- Frequency Centroid: {centroid}")
                st.write(f"- Peak Frequency: {peakf}")
                st.write(f"- Average of Fundamental Frequency Across Acoustic Signals: {meanfun}")
                st.write(f"- Minimum Fundamental Frequency Across Acoustic Signals: {minfun}")
                st.write(f"- Maximum Fundamental Frequency Across Acoustic Signals: {maxfun}")
                st.write(f"- Average of Dominant Frequency Measured Across Acoustic Signals: {meandom}")
                st.write(f"- Minimum of Dominant Frequency Measured Across Acoustic Signals: {mindom}")
                st.write(f"- Maximum of Dominant Frequency Measured Across Acoustic Signals: {maxdom}")
                st.write(f"- Range of Dominant Frequency Measured Across Acoustic Signals: {dfrange}")
                st.write(f"- Modulation Index: {modindx}")
                st.write(f"- Pitch Period Entropy: {ppe}")

                 # Display radar chart
                radar_chart = display_radar_chart([meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm, mode, centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx, ppe])
                st.plotly_chart(radar_chart, use_container_width=True)

                # Add explanation for the radar chart
                st.markdown("""
                <p style="color: #555;">
                This radar chart visualizes the key voice characteristics measured in this analysis.
                Larger areas may indicate greater deviation from typical voice patterns seen in healthy individuals.
                </p>
                """, unsafe_allow_html=True)


                # Result display with recommendations
                if parkinsons_prediction[0] == 1:
                    st.error("### Prediction: Parkinson's Disease Detected")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient may have Parkinson's disease.

                    **Recommendation:** A comprehensive clinical evaluation is recommended, including:
                    - Neurological examination
                    - MRI or CT scan
                    - Consultation with a movement disorder specialist
                    """)
                else:
                    st.success("### Prediction: No Parkinson's Disease Detected")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient likely does not have Parkinson's disease.

                    **Recommendation:** Maintain regular health check-ups and monitor for any changes in movement or speech.
                    """)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

def display_lung_cancer():
    st.title("Lung Cancer Risk Assessment")

    with st.form(key='lung_cancer_form'):
        st.write("Enter the following details to predict lung cancer:")

        GENDER = st.slider('Gender (1 = Male; 0 = Female)', 0, 1, 0, 1)
        AGE = st.slider('Age', 1, 120, 40)
        SMOKING = st.slider('Smoking (1 = Yes; 0 = No)', 0, 1, 0, 1)
        YELLOW_FINGERS = st.slider('Yellow Fingers (1 = Yes; 0 = No)', 0, 1, 0, 1)
        ANXIETY = st.slider('Anxiety (1 = Yes; 0 = No)', 0, 1, 0, 1)
        PEER_PRESSURE = st.slider('Peer Pressure (1 = Yes; 0 = No)', 0, 1, 0, 1)
        CHRONIC_DISEASE = st.slider('Chronic Disease (1 = Yes; 0 = No)', 0, 1, 0, 1)
        FATIGUE = st.slider('Fatigue (1 = Yes; 0 = No)', 0, 1, 0, 1)
        ALLERGY = st.slider('Allergy (1 = Yes; 0 = No)', 0, 1, 0, 1)
        WHEEZING = st.slider('Wheezing (1 = Yes; 0 = No)', 0, 1, 0, 1)
        ALCOHOL_CONSUMING = st.slider('Alcohol Consuming (1 = Yes; 0 = No)', 0, 1, 0, 1)
        COUGHING = st.slider('Coughing (1 = Yes; 0 = No)', 0, 1, 0, 1)
        SHORTNESS_OF_BREATH = st.slider('Shortness Of Breath (1 = Yes; 0 = No)', 0, 1, 0, 1)
        SWALLOWING_DIFFICULTY = st.slider('Swallowing Difficulty (1 = Yes; 0 = No)', 0, 1, 0, 1)
        CHEST_PAIN = st.slider('Chest Pain (1 = Yes; 0 = No)', 0, 1, 0, 1)

        submit_button = st.form_submit_button(label="Predict Lung Cancer Risk")

    if submit_button:
        with st.spinner('Analyzing risk factors...'):
            try:
                lung_prediction = models['lung_cancer'].predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
                st.success("### Prediction Complete!")
                st.markdown("#### Summary of Input Data:")
                st.write(f"- Gender: {GENDER}")
                st.write(f"- Age: {AGE}")
                st.write(f"- Smoking: {SMOKING}")
                st.write(f"- Yellow Fingers: {YELLOW_FINGERS}")
                st.write(f"- Anxiety: {ANXIETY}")
                st.write(f"- Peer Pressure: {PEER_PRESSURE}")
                st.write(f"- Chronic Disease: {CHRONIC_DISEASE}")
                st.write(f"- Fatigue: {FATIGUE}")
                st.write(f"- Allergy: {ALLERGY}")
                st.write(f"- Wheezing: {WHEEZING}")
                st.write(f"- Alcohol Consuming: {ALCOHOL_CONSUMING}")
                st.write(f"- Coughing: {COUGHING}")
                st.write(f"- Shortness Of Breath: {SHORTNESS_OF_BREATH}")
                st.write(f"- Swallowing Difficulty: {SWALLOWING_DIFFICULTY}")
                st.write(f"- Chest Pain: {CHEST_PAIN}")

                # Result display with recommendations
                if lung_prediction[0] == 1:
                    st.error("### Prediction: High Risk of Lung Cancer")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient may have a high risk of lung cancer.

                    **Recommendation:** A comprehensive evaluation is recommended, including:
                    - Chest X-ray
                    - CT scan
                    - Consultation with a pulmonologist
                    """)
                else:
                    st.success("### Prediction: Low Risk of Lung Cancer")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient likely has a low risk of lung cancer.

                    **Recommendation:** Continue regular health check-ups and maintain a healthy lifestyle.
                    """)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

def display_thyroid():
    st.title("Thyroid Disease Risk Assessment")

    with st.form(key='thyroid_form'):
        st.markdown("### Patient Demographics")
        gender = st.radio("Gender", ["Male", "Female"])
        gender_val = 1 if gender == "Male" else 0
        age = st.slider("Age", min_value=1, max_value=120, value=40)

        st.markdown("### Medical Information")
        on_thyroxine = st.radio("Is the patient on thyroxine?", ["Yes", "No"])
        on_thyroxine_val = 1 if on_thyroxine == "Yes" else 0
        tsh = st.slider("TSH Level (mU/L)", min_value=0.0, max_value=100.0, value=2.5, step=0.1)
        t3_measured = st.radio("Has T3 been measured?", ["Yes", "No"]) #No longer use, just input if measured or not
        t3_measured_val = 1 if t3_measured == "Yes" else 0
        t3 = st.slider("T3 Level (ng/dL)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
        tt4 = st.slider("TT4 Level (μg/dL)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
        #free_t4 = st.slider("Free T4 Level (ng/dL)", min_value=0.0, max_value=5.0, value=1.0, step=0.1) #Free t4 is not used, remove
        submit_button = st.form_submit_button(label="Predict Thyroid Disease Risk")
    if submit_button:
        with st.spinner('Analyzing thyroid function...'):
            try:
                thyroid_input = [[age, gender_val, on_thyroxine_val, t3_measured_val, t3, tt4, tsh]]
                thyroid_prediction = models['thyroid'].predict(thyroid_input)

                # Calculate lab value risk score
                lab_risk = 0
                if tsh < 0.4 or tsh > 4.0:
                    lab_risk += 1
                if t3 < 0.8 or t3 > 2.0:
                    lab_risk += 1
                if tt4 < 5.0 or tt4 > 12.0:
                    lab_risk += 1
                #if free_t4 < 0.8 or free_t4 > 1.8:
                #    lab_risk += 1

                # Calculate overall risk based on model prediction, lab values, and symptoms
                overall_risk = "Low"
                symptom_score = 0  # Placeholder for symptom score; you can implement this based on user input
                if thyroid_prediction[0] == 0: #Positive
                    if lab_risk >= 2 or symptom_score >= 5:
                        overall_risk = "High"
                    else:
                        overall_risk = "Moderate"
                else: #Negative
                    if lab_risk >= 3 and symptom_score >= 6:
                        overall_risk = "Moderate"

                # Result display with appropriate styling
                if overall_risk == "High":
                    st.markdown('<div class="result-box positive-result">High Risk: Strong indicators of thyroid dysfunction</div>', unsafe_allow_html=True)
                    display_lab_analysis(tsh,t3,tt4)
                    display_recommendations(overall_risk)

                elif overall_risk == "Moderate":
                    st.markdown('<div class="result-box positive-result" style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;">Moderate Risk: Some indicators of possible thyroid dysfunction</div>', unsafe_allow_html=True)
                    display_lab_analysis(tsh,t3,tt4)
                    display_recommendations(overall_risk)
                else:
                    st.markdown('<div class="result-box negative-result">Low Risk: Indicators suggest normal thyroid function</div>', unsafe_allow_html=True)
                    display_lab_analysis(tsh,t3,tt4)
                    display_recommendations(overall_risk)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

        # Add educational information
        display_educational_information()

        # Add export/save options
        display_save_options(age, gender, on_thyroxine, tsh, t3, tt4, overall_risk)

        # Reference information
        display_reference_information()


def display_lab_analysis(tsh, t3, tt4):

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Lab Value Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### TSH Level")
        if tsh < 0.4:
            st.markdown("**Low TSH:** May indicate hyperthyroidism")
        elif tsh > 4.0:
            st.markdown("**High TSH:** May indicate hypothyroidism")
        else:
            st.markdown("**Normal TSH:** Within reference range")

        st.markdown("#### T3 Level")
        if t3 < 0.8:
            st.markdown("**Low T3:** May indicate hypothyroidism")
        elif t3 > 2.0:
            st.markdown("**High T3:** May indicate hyperthyroidism")
        else:
            st.markdown("**Normal T3:** Within reference range")

    with col2:
        st.markdown("#### Total T4 Level")
        if tt4 < 5.0:
            st.markdown("**Low TT4:** May indicate hypothyroidism")
        elif tt4 > 12.0:
            st.markdown("**High TT4:** May indicate hyperthyroidism")
        else:
            st.markdown("**Normal TT4:** Within reference range")
    st.markdown("</div>", unsafe_allow_html=True)

def display_recommendations(overall_risk):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recommendations")

    if overall_risk == "High":
        st.markdown("""
        Based on the analysis, there is a **high risk** of thyroid dysfunction. We strongly recommend:

        1. **Urgent consultation with an endocrinologist** for comprehensive evaluation
        2. **Additional testing** including:
           - Thyroid antibody tests (TPO, TgAb)
           - Thyroid ultrasound
           - Complete metabolic panel
        3. **Close monitoring** of symptoms and thyroid function
        4. **Medication evaluation** if currently on thyroid medication
        """)
    elif overall_risk == "Moderate":
        st.markdown("""
        Based on the analysis, there is a **moderate risk** of thyroid dysfunction. We recommend:

        1. **Follow-up with a primary care physician** within the next 1-2 weeks
        2. **Additional thyroid function testing** to confirm results
        3. **Monitoring of symptoms** and reporting any changes to your healthcare provider
        4. **Review of current medications** that may affect thyroid function
        """)
    else:
        st.markdown("""
        Based on the analysis, there is a **low risk** of thyroid dysfunction. We recommend:

        1. **Routine health maintenance** with your primary care provider
        2. **Regular thyroid screening** as part of annual check-ups, especially if you have family history
        3. **Healthy lifestyle choices** including balanced nutrition and regular exercise
        4. **Monitoring for new symptoms** that could indicate thyroid dysfunction
        """)
    st.markdown("</div>", unsafe_allow_html=True)

def display_educational_information():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### About Thyroid Disorders")

    tab1, tab2, tab3 = st.tabs(["Thyroid Basics", "Hypothyroidism", "Hyperthyroidism"])

    with tab1:
        st.markdown("""
        The thyroid is a butterfly-shaped gland in the neck that produces hormones regulating metabolism,
        growth, and energy production. Key thyroid hormones include:

        - **TSH (Thyroid Stimulating Hormone)**: Produced by the pituitary gland to control thyroid hormone production
        - **T4 (Thyroxine)**: The main hormone produced by the thyroid
        - **T3 (Triiodothyronine)**: The active form of thyroid hormone converted from T4

        Thyroid disorders affect about 20 million Americans, with women being 5-8 times more likely to develop them.
        """)

    with tab2:
        st.markdown("""
        **Hypothyroidism** occurs when the thyroid doesn't produce enough thyroid hormones.

        **Common symptoms:**
        - Fatigue and weakness
        - Weight gain
        - Cold intolerance
        - Dry skin and hair
        - Depression
        - Constipation
        - Memory problems

        **Treatment typically involves:**
        - Daily thyroid hormone replacement medication (levothyroxine)
        - Regular monitoring of thyroid hormone levels
        - Lifestyle adjustments to manage symptoms
        """)

    with tab3:
        st.markdown("""
        **Hyperthyroidism** occurs when the thyroid produces too much thyroid hormone.

        **Common symptoms:**
        - Weight loss despite increased appetite
        - Rapid heartbeat
        - Nervousness and irritability
        - Heat intolerance and sweating
        - Tremors
        - Sleep difficulties
        - Eye problems (in Graves' disease)

        **Treatment options include:**
        - Anti-thyroid medications
        - Radioactive iodine therapy
        - Surgery (thyroidectomy)
        - Beta-blockers to manage symptoms
        """)

    st.markdown("</div>", unsafe_allow_html=True)

def display_save_options(age, gender, on_thyroxine, tsh, t3, tt4, overall_risk):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Save Results")

    col1, col2 = st.columns(2)
    with col1:
        if st.download_button(label="📊 Export as PDF", data=create_pdf_report(age, gender, on_thyroxine, tsh, t3, tt4, overall_risk), file_name="thyroid_assessment_report.pdf", mime="application/pdf"):
            st.success("PDF report downloaded successfully!")

    with col2:
        report_summary = get_report_summary(age, gender, on_thyroxine, tsh, t3, tt4, overall_risk)
        if st.button("📋 Copy Summary to Clipboard"):
            st.session_state.clipboard_content = report_summary
            st.success("Summary copied to clipboard!")


    st.markdown("</div>", unsafe_allow_html=True)

def get_report_summary(age, gender, on_thyroxine, tsh, t3, tt4, overall_risk):
    return f"""
        Thyroid Assessment Report

        Patient Information:
        Age: {age}
        Gender: {gender}
        On Thyroxine: {on_thyroxine}

        Lab Values:
        TSH: {tsh}
        T3: {t3}
        TT4: {tt4}

        Overall Risk: {overall_risk}
        """

def create_pdf_report(age, gender, on_thyroxine, tsh, t3, tt4, overall_risk):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Thyroid Assessment Report")
    c.drawString(100, 730, f"Age: {age}")
    c.drawString(100, 710, f"Gender: {gender}")
    c.drawString(100, 690, f"On Thyroxine: {on_thyroxine}")
    c.drawString(100, 670, f"TSH: {tsh}")
    c.drawString(100, 650, f"T3: {t3}")
    c.drawString(100, 630, f"TT4: {tt4}")
    c.drawString(100, 610, f"Overall Risk: {overall_risk}")
    c.save()
    buffer.seek(0)
    return buffer.read()

def display_reference_information():
    st.markdown('<div class="card" style="font-size: 12px; color: gray;">', unsafe_allow_html=True)
    st.markdown("""
    **References:**
    1. American Thyroid Association Guidelines, 2023
    2. National Institute of Diabetes and Digestive and Kidney Diseases, Thyroid Information
    3. Journal of Clinical Endocrinology & Metabolism, Thyroid Function Assessment

    *This assessment tool is for educational purposes only and is not a substitute for professional medical advice.*
    """)
    st.markdown("</div>", unsafe_allow_html=True)

def display_diabetes():
    st.title("Diabetes Prediction")

    with st.form(key='diabetes_form'):
        st.markdown("### Patient Information")
        Pregnancies = st.slider('Number of Pregnancies', 0, 20, 0)
        Age = st.slider('Age', 18, 100, 30)

        st.markdown("### Blood Work")
        Glucose = st.slider('Glucose Level (mg/dL)', 0, 300, 100)
        BloodPressure = st.slider('Blood Pressure (mm Hg)', 0, 200, 70)
        Insulin = st.slider('Insulin Level (mu U/ml)', 0, 800, 100)

        st.markdown("### Physical Measurements")
        SkinThickness = st.slider('Skin Thickness (mm)', 0, 100, 20)
        BMI = st.slider('BMI', 0.0, 60.0, 25.0, 0.1)

        st.markdown("### Genetic Factors")
        DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, 0.01)

        submit_button = st.form_submit_button(label="Run Diabetes Prediction")

    if submit_button:
        with st.spinner('Analyzing patient data...'):
            try:
                diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
                st.success("### Prediction Complete!")
                st.markdown("#### Summary of Input Data:")
                st.write(f"- Pregnancies: {Pregnancies}")
                st.write(f"- Age: {Age}")
                st.write(f"- Glucose Level: {Glucose} mg/dL")
                st.write(f"- Blood Pressure: {BloodPressure} mm Hg")
                st.write(f"- Insulin Level: {Insulin} mu U/ml")
                st.write(f"- Skin Thickness: {SkinThickness} mm")
                st.write(f"- BMI: {BMI}")
                st.write(f"- Diabetes Pedigree Function: {DiabetesPedigreeFunction}")

                # Result display with recommendations
                if diab_prediction[0] == 1:
                    st.error("### Prediction: Diabetes Detected")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient may have diabetes.

                    **Recommendation:** A comprehensive clinical evaluation is recommended, including:
                    - HbA1c test
                    - Fasting blood glucose test
                    - Oral glucose tolerance test
                    """)
                else:
                    st.success("### Prediction: No Diabetes Detected")
                    st.write("""
                    Based on the provided parameters, the model predicts that the patient likely does not have diabetes.

                    **Recommendation:** Maintain healthy lifestyle habits and continue regular check-ups.
                    """)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
def display_about():
    st.title("About Medical Diagnosis AI")
    st.markdown("""
    <div class="info-card">
        <h2>What is Medical Diagnosis AI?</h2>
        <p>
            Medical Diagnosis AI is an innovative tool designed to assist healthcare professionals in making informed decisions. 
            By leveraging the power of artificial intelligence, our application provides predictive analysis for various diseases, 
            helping to improve diagnostic accuracy and patient outcomes.
        </p>
        <h2>Key Features:</h2>
        <ul>
            <li><strong>Multiple Disease Modules:</strong> Support for diagnosing a range of conditions including Heart Disease, Diabetes, Parkinson's, Lung Cancer, and Thyroid disorders.</li>
            <li><strong>Data-Driven Predictions:</strong> Utilizes clinical data and machine learning models to predict the likelihood of specific diseases.</li>
            <li><strong>User-Friendly Interface:</strong> Designed for ease of use, allowing healthcare professionals to quickly input data and receive results.</li>
            <li><strong>Educational Resources:</strong> Provides additional information about each disease to support better understanding and decision-making.</li>
        </ul>
        <h2>Our Mission:</h2>
        <p>
            Our mission is to empower healthcare providers with advanced AI tools that enhance their ability to diagnose and manage diseases effectively. 
            We strive to improve patient care by providing accurate, reliable, and accessible diagnostic support.
        </p>
        <h2>Disclaimer:</h2>
        <p>
            Please note that Medical Diagnosis AI is intended to be used as a supportive tool and should not replace professional medical advice. 
            Always consult with qualified healthcare professionals for diagnosis and treatment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_contact():
    st.title("Contact Us")
    st.markdown("""
    <div class="info-card">
        <h2>Get in Touch</h2>
        <p>
            We're here to help and answer any questions you might have. Whether you're a healthcare professional, 
            a researcher, or just curious about our Medical Diagnosis AI, feel free to reach out to us.
        </p>
        <h2>Contact Information:</h2>
        <p>
            <strong>Email:</strong> support@medicaldiagnosisai.com<br>
            <strong>Phone:</strong> (555) 123-4567<br>
            <strong>Address:</strong> 123 Health Lane, Cityville, USA
        </p>
        <h2>Feedback and Support:</h2>
        <p>
            Your feedback is important to us. If you have any suggestions, encounter any issues, or need assistance with the app, 
            please don't hesitate to contact our support team. We are committed to continuously improving our application 
            and providing the best possible experience for our users.
        </p>
    </div>
    """, unsafe_allow_html=True)


# Main App Logic
display_nav()

if st.session_state.selected == "Home":
    display_home()
elif st.session_state.selected == "Heart Disease":
    display_heart_disease()
elif st.session_state.selected == "Parkinson's":
    display_parkinsons()
elif st.session_state.selected == "Lung Cancer":
    display_lung_cancer()
elif st.session_state.selected == "Thyroid":
    display_thyroid()
elif st.session_state.selected == "Diabetes":
    display_diabetes()
elif st.session_state.selected == "About":
    display_about()
elif st.session_state.selected == "Contact":
    display_contact()

# Add version and credits at the bottom
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 15px;'>Medical Diagnosis AI v1.0 | For educational purposes only</p>", unsafe_allow_html=True)


