import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Lung Diagnostics AI", page_icon="🩺", layout="wide")

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8f9fa; color: #212529; }
    
    /* Login Box */
    .login-box {
        padding: 40px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-top: 4px solid #0d6efd;
    }
    
    /* Blue Button */
    div.stButton > button {
        background-color: #0d6efd;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #0b5ed7;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #dee2e6; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ""

# --- CREDENTIALS ---
USERS = { "Anika": "millenium", "Harsha": "millenium", "Vamsi": "millenium" }

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('lung_disease_model.h5')
    return model

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ==========================================
# 1. LOGIN PAGE
# ==========================================
def login_page():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🏥 Secure PACS Login")
        st.info("Authorized Personnel Only")
        
        with st.form("login_form"):
            username = st.text_input("UserID")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Authenticate")
            
            if submit_button:
                if username in USERS and USERS[username] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = username
                    st.success("Access Granted.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid Credentials.")

# ==========================================
# 2. MAIN DASHBOARD
# ==========================================
def main_app():
    model = load_model()
    
    # --- HEADER ---
    st.title("🫁 Pulmonary Diagnostic Unit")
    # CHANGED HERE: Removed "Dr." and changed "Operator" to "User"
    st.markdown(f"**User:** {st.session_state['user_name']} | **Status:** Online")
    st.markdown("---")

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Control Panel")
        st.write("### AI Calibration")
        confidence_threshold = st.slider("Diagnostic Threshold", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"Current Sensitivity: {confidence_threshold}")
        
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- MAIN LAYOUT (Split View) ---
    col_left, col_right = st.columns([1, 1], gap="medium")

    # LEFT: IMAGE INPUT
    with col_left:
        st.subheader("1. Patient Radiograph")
        file = st.file_uploader("Select DICOM/JPEG Image", type=["jpg", "png", "jpeg"])
        
        if file is not None:
            image = Image.open(file)
            st.image(image, caption="Current Scan", use_container_width=True)
        else:
            st.info("Waiting for patient data...")

    # RIGHT: DIAGNOSIS
    with col_right:
        st.subheader("2. Diagnostic Report")
        
        if file is not None:
            st.write("Image loaded. Ready for analysis.")
            
            if st.button("▶ GENERATE REPORT", type="primary"):
                
                # Logic
                def import_and_predict(image_data, model):
                    size = (150, 150)
                    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
                    img = np.asarray(image)
                    
                    if img.ndim == 2: 
                        img = np.stack((img,)*3, axis=-1)
                    else: 
                        img = img[:,:,:3]
                    
                    img = img / 255.0
                    img_reshape = np.expand_dims(img, axis=0)
                    return model.predict(img_reshape)

                with st.spinner('Analyzing...'):
                    predictions = import_and_predict(image, model)
                    score = predictions[0][0]
                    
                    # --- RESULTS DISPLAY ---
                    if score > confidence_threshold:
                        # PNEUMONIA (Red Box)
                        st.error("DETECTED: PNEUMONIA")
                        st.markdown(f"""
                            <div style="background-color:#fff3cd; padding:15px; border-radius:5px; border:1px solid #ffecb5;">
                                <h3 style="color:#856404; margin:0;">⚠️ POSITIVE</h3>
                                <p style="margin:5px 0 0 0;"><strong>Confidence:</strong> {(score*100):.2f}%</p>
                                <p style="font-size:14px;">Recommendation: Clinical correlation required.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.toast("Alert: Pathology Detected", icon="⚠️")
                        
                    else:
                        # NORMAL (Green Box)
                        st.success("RESULT: NORMAL")
                        st.markdown(f"""
                            <div style="background-color:#d4edda; padding:15px; border-radius:5px; border:1px solid #c3e6cb;">
                                <h3 style="color:#155724; margin:0;">✅ NEGATIVE</h3>
                                <p style="margin:5px 0 0 0;"><strong>Confidence:</strong> {((1-score)*100):.2f}%</p>
                                <p style="font-size:14px;">Status: No abnormalities detected.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.toast("Analysis Complete: Normal", icon="✅")

        else:
            st.markdown("""
                <div style="padding:20px; border:2px dashed #ccc; border-radius:10px; text-align:center; color:#666;">
                    Upload an X-ray on the left to begin diagnosis.
                </div>
            """, unsafe_allow_html=True)

# --- RUN APP ---
if st.session_state['logged_in']:
    main_app()
else:
    login_page()