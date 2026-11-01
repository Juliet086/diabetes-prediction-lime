import streamlit as st # type: ignore
st.set_page_config(layout="wide")


def inject_global_button_css():
    button_css = """
    <style>
    .stButton>button {
        background-color: #AFEEEE !important;
        color: #FFFFFF !important;
        padding: 10px 24px !important;
        border: 5px #87CEEB !important;
        border-radius: 7px !important;
        text-align: center !important;
        text-decoration: none !important;
        display: inline-block !important;
        font-size: 16px !important;
        margin: 4px 2px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #87CEEB !important;
        color: #FFFFFF !important;
    }
    .stButton>button:active {
        background-color: #87CEEB !important;
        color: #FFFFFF !important;
        outline: none !important;
    }

    /* Style for input labels */
    label {
        color: #66CDAA !important;
    }

    /* Style for input widgets */
    .stNumberInput > label,
    .stTextInput > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stRadio > label,
    .stCheckbox > label,
    .stSlider > label,
    .stDateInput > label,
    .stTimeInput > label,
    .stColorPicker > label {
        color: white !important;
    }

    /* Style for the input boxes themselves */
    input[type="number"],
    input[type="text"],
    input[type="date"],
    input[type="time"],
    input[type="color"],
    select,
    textarea {
        background-color: #AFEEEE !important;
        color: black !important;
        border-radius: 3px !important;
        padding: 5px !important;
    }

    /* Style for the "+" and "-" buttons in number input */
    .streamlit-number-input button {
        color: black !important;
        background-color: #66CDAA !important;
        border-radius: 3px !important;
    }

    /* Style for the prediction output text and general paragraphs */
    div[data-testid="stMarkdownContainer"] p {
        color: black !important;
    }

    /* Force headings (h1, h2, h3, etc.) to be dark/black so they are visible on a light background */
    h1, h2, h3, h4, h5, h6 {
        color: #2C3E50 !important;
    }

    /* Style for bordered containers */
    .st-b7 {
        border: 1px solid #66CDAA !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }

    /* ---> PASTE THE TABLE STYLING HERE <--- */
    table {
        width: 100%;
        border-collapse: collapse;
        background-color: white;
        color: black;
    }
    th {
        background-color: #66CDAA !important;
        color: white !important;
        padding: 8px;
        text-align: left;
        border: 1px solid #2C3E50;
    }
    td {
        padding: 8px;
        border: 1px solid #ddd;
        color: black;
    }
    </style>
    """
    st.markdown(button_css, unsafe_allow_html=True)

# Set your desired background color for the app
app_background_color = "#E0FFFF"

# FIX: Changed 'def"""' to 'f"""' to fix the syntax error
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {app_background_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Inject the button and input field CSS
inject_global_button_css()

import home
import prediction
import evaluation
import insights

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

# Display the current page
if st.session_state['current_page'] == 'home':
    home.app()
elif st.session_state['current_page'] == 'prediction':
    prediction.app()
elif st.session_state['current_page'] == 'evaluation':
    evaluation.app()
elif st.session_state['current_page'] == 'insights':
    insights.app()