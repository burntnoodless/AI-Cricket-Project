import plotly.graph_objects as go
import streamlit as st
import tempfile
import cv2
import pandas as pd
from analysis import analyze_cricket_shot
from advice_engine import CricketAdviceEngine, ImprovementEngine

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="CricketSense AI Coach",
    page_icon="🏏",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'final_metrics' not in st.session_state:
    st.session_state.final_metrics = None
if 'analyzed_video_path' not in st.session_state:
    st.session_state.analyzed_video_path = None
if 'original_metrics' not in st.session_state:
    st.session_state.original_metrics = None
if 'original_advice' not in st.session_state:
    st.session_state.original_advice = None
if 'followup_metrics' not in st.session_state:
    st.session_state.followup_metrics = None
if 'improvement_analysis' not in st.session_state:
    st.session_state.improvement_analysis = None

# Theme toggle function
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Apple-inspired CSS with dark/light mode
def get_css(theme='light'):
    if theme == 'dark':
        return """
        <style>
            /* Dark Mode Styling */
            .main {
                background-color: #000000 !important;
            }

            .stApp {
                background-color: #000000 !important;
            }

            /* Streamlit Text Elements - Dark Mode */
            .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
                color: #f5f5f7 !important;
            }

            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #f5f5f7 !important;
            }

            /* File Uploader - Dark Mode */
            .stFileUploader label {
                color: #f5f5f7 !important;
            }

            .stFileUploader div[data-testid="stFileUploaderDropzone"] {
                background-color: #1d1d1f !important;
                border-color: #3d3d3d !important;
            }

            .stFileUploader div[data-testid="stFileUploaderDropzone"] p {
                color: #86868b !important;
            }

            /* Progress Bar - Dark Mode */
            .stProgress > div > div {
                background-color: #0071e3 !important;
            }

            /* Info/Warning boxes - Dark Mode */
            .stAlert {
                background-color: #1d1d1f !important;
                color: #f5f5f7 !important;
            }

            /* Hero Section */
            .hero-section {
                text-align: center;
                padding: 8rem 2rem 6rem 2rem;
                background: #000000;
            }

            .hero-title {
                font-size: 5rem;
                font-weight: 700;
                color: #ffffff !important;
                margin-bottom: 1.5rem;
                letter-spacing: -0.02em;
                line-height: 1.1;
            }

            .hero-subtitle {
                font-size: 1.75rem;
                color: #86868b !important;
                font-weight: 400;
                margin-bottom: 3rem;
                line-height: 1.4;
            }

            /* Feature Cards */
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                padding: 4rem 2rem;
                max-width: 1200px;
                margin: 0 auto;
            }

            .feature-card {
                background: #1d1d1f;
                padding: 3rem 2rem;
                border-radius: 18px;
                text-align: center;
                transition: transform 0.3s ease;
            }

            .feature-card:hover {
                transform: translateY(-8px);
            }

            .feature-icon {
                font-size: 3.5rem;
                margin-bottom: 1.5rem;
            }

            .feature-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: #f5f5f7 !important;
                margin-bottom: 1rem;
            }

            .feature-description {
                font-size: 1rem;
                color: #86868b !important;
                line-height: 1.6;
            }

            /* Analysis Page Styling */
            .section-header {
                font-size: 2rem;
                font-weight: 600;
                color: #f5f5f7 !important;
                margin-bottom: 2rem;
                letter-spacing: -0.01em;
            }

            .metric-card {
                background: #1d1d1f;
                padding: 2rem;
                border-radius: 18px;
                margin-bottom: 1.5rem;
                transition: transform 0.2s ease;
            }

            .metric-card:hover {
                transform: translateY(-4px);
            }

            .metric-label {
                font-size: 0.9rem;
                color: #86868b !important;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }

            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: #f5f5f7 !important;
            }

            .shot-type-badge {
                display: inline-block;
                background: #0071e3;
                color: white !important;
                padding: 0.75rem 2rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1.1rem;
                margin: 1rem 0;
            }

            /* Advice Cards */
            .advice-strength {
                background: #1d3a1f;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid #30d158;
                color: #f5f5f7 !important;
            }

            .advice-flaw {
                background: #3a2817;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid #ff9f0a;
                color: #f5f5f7 !important;
            }

            .advice-recommendation {
                background: #1a2a3a;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid #0a84ff;
                color: #f5f5f7 !important;
            }

            /* Header */
            .app-logo {
                font-size: 1.5rem;
                font-weight: 600;
                color: #f5f5f7 !important;
            }

            /* Video Container */
            .video-container {
                background: #1d1d1f;
                padding: 1.5rem;
                border-radius: 18px;
                margin-bottom: 2rem;
            }

            /* Performance Score */
            .performance-score-card {
                background: linear-gradient(135deg, #0a84ff 0%, #0071e3 100%);
                padding: 2.5rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .performance-score-value {
                font-size: 4rem;
                font-weight: 700;
                margin: 1rem 0;
                color: white !important;
            }

            /* Buttons - Dark Mode - Comprehensive */
            .stButton > button,
            .stButton button,
            button[kind="primary"],
            button[kind="secondary"],
            div[data-testid="stButton"] > button {
                background-color: #0071e3 !important;
                background: #0071e3 !important;
                color: white !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
                padding: 0.75rem 2rem !important;
                border-radius: 980px !important;
                border: none !important;
                transition: all 0.3s ease !important;
                opacity: 1 !important;
                visibility: visible !important;
            }

            .stButton > button:hover,
            .stButton button:hover,
            button[kind="primary"]:hover,
            button[kind="secondary"]:hover,
            div[data-testid="stButton"] > button:hover {
                background-color: #0077ed !important;
                background: #0077ed !important;
                transform: scale(1.02) !important;
                color: white !important;
            }

            .stButton > button:focus,
            .stButton > button:active,
            .stButton button:focus,
            .stButton button:active {
                background-color: #0071e3 !important;
                background: #0071e3 !important;
                color: white !important;
                border: none !important;
                box-shadow: none !important;
            }

            .stButton > button:focus:not(:active) {
                background-color: #0071e3 !important;
                color: white !important;
            }

            /* Download Button - Dark Mode */
            .stDownloadButton > button,
            .stDownloadButton button,
            div[data-testid="stDownloadButton"] > button {
                background-color: #0071e3 !important;
                background: #0071e3 !important;
                color: white !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
                padding: 0.75rem 2rem !important;
                border-radius: 980px !important;
                border: none !important;
                opacity: 1 !important;
                visibility: visible !important;
            }

            .stDownloadButton > button:hover,
            .stDownloadButton button:hover {
                background-color: #0077ed !important;
                background: #0077ed !important;
                color: white !important;
            }

            .stDownloadButton > button:focus,
            .stDownloadButton > button:active {
                background-color: #0071e3 !important;
                color: white !important;
            }

            /* Tabs - Dark Mode */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
                background-color: transparent;
            }

            .stTabs [data-baseweb="tab"] {
                background-color: transparent;
                color: #86868b !important;
                font-weight: 500;
                border: none;
                padding: 1rem 0;
            }

            .stTabs [aria-selected="true"] {
                color: #0a84ff !important;
                border-bottom: 2px solid #0a84ff;
            }

            /* Code blocks - Dark Mode */
            .stCodeBlock {
                background-color: #1d1d1f !important;
            }

            code {
                color: #f5f5f7 !important;
            }

            /* Ensure all button types are visible - Dark Mode */
            button {
                opacity: 1 !important;
                visibility: visible !important;
            }

            /* Fix for button text visibility */
            button p, button span, button div {
                color: white !important;
            }

            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            /* Improvement Comparison Styles - Dark Mode */
            .improvement-header {
                text-align: center;
                padding: 2rem;
                margin-bottom: 2rem;
            }

            .comparison-card {
                background: #1d1d1f;
                padding: 1.5rem;
                border-radius: 18px;
                margin-bottom: 1.5rem;
            }

            .metric-comparison {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                background: #2d2d2f;
                border-radius: 12px;
                margin-bottom: 0.75rem;
            }

            .verdict-improved {
                background: linear-gradient(135deg, #30d158 0%, #28a745 100%);
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .verdict-regressed {
                background: linear-gradient(135deg, #ff453a 0%, #dc3545 100%);
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .verdict-maintained {
                background: linear-gradient(135deg, #ff9f0a 0%, #fd7e14 100%);
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .accuracy-comparison {
                display: flex;
                justify-content: space-around;
                gap: 2rem;
                margin: 2rem 0;
            }

            .accuracy-box {
                background: #1d1d1f;
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                flex: 1;
            }

            .accuracy-box.original {
                border: 2px solid #86868b;
            }

            .accuracy-box.followup {
                border: 2px solid #0a84ff;
            }

            .accuracy-value {
                font-size: 3rem;
                font-weight: 700;
                color: #f5f5f7 !important;
            }

            .improvement-badge-positive {
                display: inline-block;
                background: #30d158;
                color: white !important;
                padding: 0.5rem 1.5rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1rem;
            }

            .improvement-badge-negative {
                display: inline-block;
                background: #ff453a;
                color: white !important;
                padding: 0.5rem 1.5rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1rem;
            }

            .improvement-badge-neutral {
                display: inline-block;
                background: #86868b;
                color: white !important;
                padding: 0.5rem 1.5rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1rem;
            }

            .metric-improved {
                background: #1d3a1f;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #30d158;
                color: #f5f5f7 !important;
            }

            .metric-regressed {
                background: #3a1d1d;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #ff453a;
                color: #f5f5f7 !important;
            }

            .metric-maintained {
                background: #2d2d2f;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #86868b;
                color: #f5f5f7 !important;
            }

            .focus-area-card {
                background: #1a2a3a;
                padding: 1.25rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #0a84ff;
                color: #f5f5f7 !important;
            }

            .drill-card {
                background: #2a1a3a;
                padding: 1.25rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #bf5af2;
                color: #f5f5f7 !important;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light Mode Styling */
            .main {
                background-color: #ffffff !important;
            }

            .stApp {
                background-color: #ffffff !important;
            }

            /* Streamlit Text Elements - Light Mode */
            .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
                color: #1d1d1f !important;
            }

            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #1d1d1f !important;
            }

            /* File Uploader - Light Mode */
            .stFileUploader label {
                color: #1d1d1f !important;
            }

            .stFileUploader div[data-testid="stFileUploaderDropzone"] {
                background-color: #f5f5f7 !important;
                border-color: #d2d2d7 !important;
            }

            .stFileUploader div[data-testid="stFileUploaderDropzone"] p {
                color: #6e6e73 !important;
            }

            /* Progress Bar - Light Mode */
            .stProgress > div > div {
                background-color: #0071e3 !important;
            }

            /* Info/Warning boxes - Light Mode */
            .stAlert {
                background-color: #f5f5f7 !important;
                color: #1d1d1f !important;
            }

            /* Hero Section */
            .hero-section {
                text-align: center;
                padding: 8rem 2rem 6rem 2rem;
                background: #ffffff;
            }

            .hero-title {
                font-size: 5rem;
                font-weight: 700;
                color: #1d1d1f !important;
                margin-bottom: 1.5rem;
                letter-spacing: -0.02em;
                line-height: 1.1;
            }

            .hero-subtitle {
                font-size: 1.75rem;
                color: #6e6e73 !important;
                font-weight: 400;
                margin-bottom: 3rem;
                line-height: 1.4;
            }

            /* Feature Cards */
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                padding: 4rem 2rem;
                max-width: 1200px;
                margin: 0 auto;
            }

            .feature-card {
                background: #f5f5f7;
                padding: 3rem 2rem;
                border-radius: 18px;
                text-align: center;
                transition: transform 0.3s ease;
            }

            .feature-card:hover {
                transform: translateY(-8px);
            }

            .feature-icon {
                font-size: 3.5rem;
                margin-bottom: 1.5rem;
            }

            .feature-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: #1d1d1f !important;
                margin-bottom: 1rem;
            }

            .feature-description {
                font-size: 1rem;
                color: #6e6e73 !important;
                line-height: 1.6;
            }

            /* Analysis Page Styling */
            .section-header {
                font-size: 2rem;
                font-weight: 600;
                color: #1d1d1f !important;
                margin-bottom: 2rem;
                letter-spacing: -0.01em;
            }

            .metric-card {
                background: #f5f5f7;
                padding: 2rem;
                border-radius: 18px;
                margin-bottom: 1.5rem;
                transition: transform 0.2s ease;
            }

            .metric-card:hover {
                transform: translateY(-4px);
            }

            .metric-label {
                font-size: 0.9rem;
                color: #6e6e73 !important;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }

            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1d1d1f !important;
            }

            .shot-type-badge {
                display: inline-block;
                background: #0071e3;
                color: white !important;
                padding: 0.75rem 2rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1.1rem;
                margin: 1rem 0;
            }

            /* Advice Cards */
            .advice-strength {
                background: #d1f2d9;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid #30d158;
                color: #1d1d1f !important;
            }

            .advice-flaw {
                background: #ffecd2;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid #ff9f0a;
                color: #1d1d1f !important;
            }

            .advice-recommendation {
                background: #d6ebff;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid #0a84ff;
                color: #1d1d1f !important;
            }

            /* Header */
            .app-logo {
                font-size: 1.5rem;
                font-weight: 600;
                color: #1d1d1f !important;
            }

            /* Video Container */
            .video-container {
                background: #f5f5f7;
                padding: 1.5rem;
                border-radius: 18px;
                margin-bottom: 2rem;
            }

            /* Performance Score */
            .performance-score-card {
                background: linear-gradient(135deg, #0a84ff 0%, #0071e3 100%);
                padding: 2.5rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .performance-score-value {
                font-size: 4rem;
                font-weight: 700;
                margin: 1rem 0;
                color: white !important;
            }

            /* Buttons - Light Mode - Comprehensive */
            .stButton > button,
            .stButton button,
            button[kind="primary"],
            button[kind="secondary"],
            div[data-testid="stButton"] > button {
                background-color: #0071e3 !important;
                background: #0071e3 !important;
                color: white !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
                padding: 0.75rem 2rem !important;
                border-radius: 980px !important;
                border: none !important;
                transition: all 0.3s ease !important;
                opacity: 1 !important;
                visibility: visible !important;
            }

            .stButton > button:hover,
            .stButton button:hover,
            button[kind="primary"]:hover,
            button[kind="secondary"]:hover,
            div[data-testid="stButton"] > button:hover {
                background-color: #0077ed !important;
                background: #0077ed !important;
                transform: scale(1.02) !important;
                color: white !important;
            }

            .stButton > button:focus,
            .stButton > button:active,
            .stButton button:focus,
            .stButton button:active {
                background-color: #0071e3 !important;
                background: #0071e3 !important;
                color: white !important;
                border: none !important;
                box-shadow: none !important;
            }

            .stButton > button:focus:not(:active) {
                background-color: #0071e3 !important;
                color: white !important;
            }

            /* Download Button - Light Mode */
            .stDownloadButton > button,
            .stDownloadButton button,
            div[data-testid="stDownloadButton"] > button {
                background-color: #0071e3 !important;
                background: #0071e3 !important;
                color: white !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
                padding: 0.75rem 2rem !important;
                border-radius: 980px !important;
                border: none !important;
                opacity: 1 !important;
                visibility: visible !important;
            }

            .stDownloadButton > button:hover,
            .stDownloadButton button:hover {
                background-color: #0077ed !important;
                background: #0077ed !important;
                color: white !important;
            }

            .stDownloadButton > button:focus,
            .stDownloadButton > button:active {
                background-color: #0071e3 !important;
                color: white !important;
            }

            /* Tabs - Light Mode */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
                background-color: transparent;
            }

            .stTabs [data-baseweb="tab"] {
                background-color: transparent;
                color: #6e6e73 !important;
                font-weight: 500;
                border: none;
                padding: 1rem 0;
            }

            .stTabs [aria-selected="true"] {
                color: #0071e3 !important;
                border-bottom: 2px solid #0071e3;
            }

            /* Code blocks - Light Mode */
            .stCodeBlock {
                background-color: #f5f5f7 !important;
            }

            code {
                color: #1d1d1f !important;
            }

            /* Ensure all button types are visible - Light Mode */
            button {
                opacity: 1 !important;
                visibility: visible !important;
            }

            /* Fix for button text visibility */
            button p, button span, button div {
                color: white !important;
            }

            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            /* Improvement Comparison Styles - Light Mode */
            .improvement-header {
                text-align: center;
                padding: 2rem;
                margin-bottom: 2rem;
            }

            .comparison-card {
                background: #f5f5f7;
                padding: 1.5rem;
                border-radius: 18px;
                margin-bottom: 1.5rem;
            }

            .metric-comparison {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                background: #ffffff;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border: 1px solid #e0e0e0;
            }

            .verdict-improved {
                background: linear-gradient(135deg, #30d158 0%, #28a745 100%);
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .verdict-regressed {
                background: linear-gradient(135deg, #ff453a 0%, #dc3545 100%);
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .verdict-maintained {
                background: linear-gradient(135deg, #ff9f0a 0%, #fd7e14 100%);
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                color: white !important;
                margin-bottom: 2rem;
            }

            .accuracy-comparison {
                display: flex;
                justify-content: space-around;
                gap: 2rem;
                margin: 2rem 0;
            }

            .accuracy-box {
                background: #ffffff;
                padding: 2rem;
                border-radius: 18px;
                text-align: center;
                flex: 1;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }

            .accuracy-box.original {
                border: 2px solid #86868b;
            }

            .accuracy-box.followup {
                border: 2px solid #0071e3;
            }

            .accuracy-value {
                font-size: 3rem;
                font-weight: 700;
                color: #1d1d1f !important;
            }

            .improvement-badge-positive {
                display: inline-block;
                background: #30d158;
                color: white !important;
                padding: 0.5rem 1.5rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1rem;
            }

            .improvement-badge-negative {
                display: inline-block;
                background: #ff453a;
                color: white !important;
                padding: 0.5rem 1.5rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1rem;
            }

            .improvement-badge-neutral {
                display: inline-block;
                background: #86868b;
                color: white !important;
                padding: 0.5rem 1.5rem;
                border-radius: 980px;
                font-weight: 600;
                font-size: 1rem;
            }

            .metric-improved {
                background: #d1f2d9;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #30d158;
                color: #1d1d1f !important;
            }

            .metric-regressed {
                background: #ffd9d6;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #ff453a;
                color: #1d1d1f !important;
            }

            .metric-maintained {
                background: #f0f0f0;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #86868b;
                color: #1d1d1f !important;
            }

            .focus-area-card {
                background: #d6ebff;
                padding: 1.25rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #0071e3;
                color: #1d1d1f !important;
            }

            .drill-card {
                background: #f0e6ff;
                padding: 1.25rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid #9b51e0;
                color: #1d1d1f !important;
            }
        </style>
        """

# Apply CSS
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# Initialize the advice engine
@st.cache_resource
def get_advice_engine():
    return CricketAdviceEngine()

advice_engine = get_advice_engine()

@st.cache_resource
def get_improvement_engine():
    return ImprovementEngine()

improvement_engine = get_improvement_engine()

def calculate_overall_performance_score(metrics):
    """Calculate overall performance score (0-100) based on key metrics."""
    scores = []

    # Elbow extension score
    if 'max_elbow_angle' in metrics:
        elbow_angle = metrics['max_elbow_angle']
        if elbow_angle >= 165:
            scores.append(100)
        elif elbow_angle >= 155:
            scores.append(80)
        elif elbow_angle >= 140:
            scores.append(60)
        else:
            scores.append(40)

    # Head stability score
    if 'head_movement' in metrics:
        head_movement = metrics['head_movement']
        if head_movement <= 8:
            scores.append(100)
        elif head_movement <= 15:
            scores.append(80)
        elif head_movement <= 25:
            scores.append(60)
        else:
            scores.append(40)

    # Body rotation score
    if 'body_rotation_total' in metrics:
        rotation = metrics['body_rotation_total']
        shot_type = metrics.get('shot_type', 'UNKNOWN')

        if shot_type in ['PULL/HOOK', 'CUT', 'SWEEP']:
            if rotation >= 45:
                scores.append(100)
            elif rotation >= 30:
                scores.append(80)
            elif rotation >= 20:
                scores.append(60)
            else:
                scores.append(40)
        elif shot_type == 'DEFENSIVE':
            if rotation <= 15:
                scores.append(100)
            elif rotation <= 25:
                scores.append(80)
            else:
                scores.append(60)
        else:
            if 20 <= rotation <= 45:
                scores.append(100)
            elif 15 <= rotation <= 60:
                scores.append(80)
            else:
                scores.append(60)

    # Weight transfer score
    if 'weight_transfer_amount' in metrics:
        transfer = metrics['weight_transfer_amount']
        shot_type = metrics.get('shot_type', 'UNKNOWN')

        if shot_type in ['DRIVE', 'FORWARD SHOT', 'LOFTED']:
            if transfer >= 8:
                scores.append(100)
            elif transfer >= 3:
                scores.append(80)
            elif transfer >= 0:
                scores.append(60)
            else:
                scores.append(40)
        elif shot_type in ['PULL/HOOK', 'CUT']:
            if transfer <= 2:
                scores.append(100)
            elif transfer <= 5:
                scores.append(80)
            else:
                scores.append(60)
        else:
            if 0 <= transfer <= 5:
                scores.append(100)
            else:
                scores.append(80)

    if not scores:
        return 0

    return round(sum(scores) / len(scores))

# Header with theme toggle
header_col1, header_col2 = st.columns([6, 1])
with header_col1:
    st.markdown('<div class="app-logo">🏏 CricketSense</div>', unsafe_allow_html=True)
with header_col2:
    theme_icon = "☀️" if st.session_state.theme == 'dark' else "🌙"
    if st.button(theme_icon, key="theme_toggle", help="Toggle dark/light mode"):
        toggle_theme()
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ==================== HOME PAGE ====================
if st.session_state.page == 'home':
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Perfect your batting.<br>One swing at a time.</h1>
        <p class="hero-subtitle">AI-powered biomechanical analysis that transforms your cricket technique</p>
    </div>
    """, unsafe_allow_html=True)

    # CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Analysis", type="primary", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()

    # Feature Cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🎥</div>
            <h3 class="feature-title">Video Analysis</h3>
            <p class="feature-description">Upload your batting video and get real-time pose detection with advanced computer vision</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Biomechanical Metrics</h3>
            <p class="feature-description">Track elbow angles, weight transfer, body rotation, and head stability with precision</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">Personalized Coaching</h3>
            <p class="feature-description">Receive shot-specific feedback, identify strengths and flaws, get actionable drills</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== ANALYSIS PAGE ====================
elif st.session_state.page == 'analysis':
    st.markdown('<p class="section-header">Upload Your Batting Video</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi"],
        help="Upload a clear video of your cricket shot"
    )

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.session_state.analyzed_video_path = video_path

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Analyze Shot", type="primary", use_container_width=True):
                final_metrics = None
                progress_bar = st.progress(0, text="Initializing analysis...")
                video_placeholder = st.empty()

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                cap.release()

                frame_generator = analyze_cricket_shot(video_path)
                frame_count = 0

                for frame, metrics in frame_generator:
                    if frame is not None:
                        with video_placeholder.container():
                            st.markdown('<div class="video-container">', unsafe_allow_html=True)
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        frame_count += 1
                        if total_frames > 0:
                            progress = frame_count / total_frames
                            progress_bar.progress(progress, text=f"Analyzing... {int(progress*100)}%")
                    if metrics:
                        final_metrics = metrics

                progress_bar.progress(1.0, text="Analysis Complete!")
                st.session_state.final_metrics = final_metrics

                # Auto-navigate to advice page
                if final_metrics:
                    st.success("Analysis complete! Taking you to results...")
                    st.session_state.page = 'advice'
                    st.rerun()

        with col2:
            if st.button("← Back to Home"):
                st.session_state.page = 'home'
                st.rerun()

# ==================== ADVICE PAGE ====================
elif st.session_state.page == 'advice':
    if st.session_state.final_metrics is None:
        st.warning("No analysis data found. Please analyze a video first.")
        if st.button("← Go to Analysis"):
            st.session_state.page = 'analysis'
            st.rerun()
    else:
        final_metrics = st.session_state.final_metrics

        # Back button
        if st.button("← Analyze Another Video"):
            st.session_state.page = 'analysis'
            st.session_state.final_metrics = None
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Your Analysis Results</p>', unsafe_allow_html=True)

        # Two column layout
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            # Shot Type
            shot_type = final_metrics.get('shot_type', 'UNKNOWN')
            shot_confidence = final_metrics.get('shot_confidence', 0.0)

            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <p class="metric-label">Detected Shot Type</p>
                    <div class="shot-type-badge">{shot_type}</div>
                    <p style="margin-top: 1rem; color: #6e6e73;">Confidence: {shot_confidence*100:.0f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # Performance Score
            performance_score = calculate_overall_performance_score(final_metrics)
            score_label = "Excellent" if performance_score >= 85 else "Good" if performance_score >= 70 else "Fair" if performance_score >= 55 else "Needs Work"

            st.markdown(f"""
                <div class="performance-score-card">
                    <p style="font-size: 1.1rem; opacity: 0.9;">Overall Performance</p>
                    <div class="performance-score-value">{performance_score}<span style="font-size: 2rem;">/100</span></div>
                    <p style="font-size: 1rem; opacity: 0.8;">{score_label}</p>
                </div>
            """, unsafe_allow_html=True)

            # Key Metrics
            st.markdown("### Key Biomechanical Metrics")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Elbow Extension</p>
                        <p class="metric-value">{final_metrics.get('max_elbow_angle', 0):.1f}°</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Body Rotation</p>
                        <p class="metric-value">{final_metrics.get('body_rotation_total', 0):.1f}°</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Weight Transfer</p>
                        <p class="metric-value">{final_metrics.get('weight_transfer_amount', 0):.1f}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Head Stability</p>
                        <p class="metric-value">{final_metrics.get('head_movement', 0):.1f}°</p>
                    </div>
                """, unsafe_allow_html=True)

            # Rotation Chart
            body_rotation_data = final_metrics.get('body_rotation_over_time', [])
            if body_rotation_data:
                st.markdown("### Body Rotation Over Time")
                df = pd.DataFrame({
                    'Frame': range(len(body_rotation_data)),
                    'Rotation (°)': body_rotation_data
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Frame'],
                    y=df['Rotation (°)'],
                    mode='lines',
                    name='Body Rotation',
                    line=dict(color='#0071e3', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 113, 227, 0.1)'
                ))

                fig.update_layout(
                    xaxis_title='Frame',
                    yaxis_title='Rotation (°)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(gridcolor='#e0e0e0', showgrid=True),
                    yaxis=dict(gridcolor='#e0e0e0', showgrid=True)
                )

                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Generate AI-powered advice
            with st.spinner("🤖 AI Coach analyzing your technique..."):
                advice = advice_engine.generate_ai_advice(final_metrics)

            st.markdown("### Personalized Coaching Advice")

            # Coach's narrative summary
            if 'shot_insights' in advice and advice['shot_insights']:
                insights = advice['shot_insights']
                if 'description' in insights and insights['description']:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; color: white;">
                        <h4 style="margin-top: 0; color: white;">🏏 Coach's Analysis</h4>
                        <p style="margin-bottom: 0; line-height: 1.6;">{insights['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Pro comparison if available
                if 'pro_comparison' in insights and insights['pro_comparison']:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                padding: 1rem; border-radius: 8px; margin-bottom: 1rem; color: white;">
                        <strong>⭐ Pro Technique Reference:</strong> {insights['pro_comparison']}
                    </div>
                    """, unsafe_allow_html=True)

            # Tabbed advice
            tab1, tab2, tab3, tab4 = st.tabs(["✅ Strengths", "⚠️ Areas to Improve", "💡 Recommendations", "📋 Full Report"])

            with tab1:
                if advice['strengths']:
                    for strength in advice['strengths']:
                        st.markdown(f'<div class="advice-strength">{strength}</div>', unsafe_allow_html=True)
                else:
                    st.info("Keep practicing to develop your strengths!")

            with tab2:
                if advice['flaws']:
                    for flaw in advice['flaws']:
                        st.markdown(f'<div class="advice-flaw">{flaw}</div>', unsafe_allow_html=True)
                else:
                    st.success("Excellent technique overall!")

            with tab3:
                if advice['recommendations']:
                    for i, rec in enumerate(advice['recommendations'], 1):
                        st.markdown(f'<div class="advice-recommendation"><strong>Drill {i}:</strong> {rec}</div>', unsafe_allow_html=True)

                if 'shot_insights' in advice and 'key_focus_areas' in advice['shot_insights']:
                    st.markdown("#### 🎯 Key Focus Areas")
                    for focus in advice['shot_insights']['key_focus_areas']:
                        st.markdown(f"- {focus}")

            with tab4:
                advice_summary = advice_engine.get_advice_summary(advice)
                st.code(advice_summary, language="text")

                st.download_button(
                    label="📥 Download Report",
                    data=advice_summary,
                    file_name="cricket_coaching_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        # Track Improvement Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3 style="margin-bottom: 1rem;">📈 Track Your Improvement</h3>
            <p style="color: #6e6e73; margin-bottom: 1.5rem;">
                Practice the advice above, then upload a new video to see how much you've improved.
                Get a side-by-side comparison with honest feedback on your progress.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Upload Follow-up Video", type="primary", use_container_width=True):
                # Store original metrics and advice for comparison
                st.session_state.original_metrics = final_metrics.copy()
                st.session_state.original_advice = advice.copy()
                st.session_state.page = 'track_improvement'
                st.rerun()

# ==================== TRACK IMPROVEMENT PAGE ====================
elif st.session_state.page == 'track_improvement':
    if st.session_state.original_metrics is None:
        st.warning("No original analysis found. Please analyze a video first.")
        if st.button("← Go to Analysis"):
            st.session_state.page = 'analysis'
            st.rerun()
    else:
        # Back button
        if st.button("← Back to Original Results"):
            st.session_state.page = 'advice'
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Upload Your Follow-up Video</p>', unsafe_allow_html=True)

        # Show original shot info
        original_shot = st.session_state.original_metrics.get('shot_type', 'UNKNOWN')
        original_score = calculate_overall_performance_score(st.session_state.original_metrics)

        st.markdown(f"""
        <div class="comparison-card">
            <p style="margin-bottom: 0.5rem;"><strong>Original Shot:</strong> {original_shot}</p>
            <p style="margin-bottom: 0;"><strong>Original Score:</strong> {original_score}/100</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 Tip: For best comparison, try to replicate the same shot type as your original video.")

        followup_file = st.file_uploader(
            "Upload your follow-up video",
            type=["mp4", "mov", "avi"],
            help="Upload a video of you practicing the same shot after following the advice",
            key="followup_uploader"
        )

        if followup_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(followup_file.read())
                followup_video_path = tfile.name

            col1, col2 = st.columns([2, 1])

            with col1:
                if st.button("Analyze Follow-up Shot", type="primary", use_container_width=True):
                    followup_metrics = None
                    progress_bar = st.progress(0, text="Analyzing your follow-up shot...")
                    video_placeholder = st.empty()

                    cap = cv2.VideoCapture(followup_video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                    cap.release()

                    frame_generator = analyze_cricket_shot(followup_video_path)
                    frame_count = 0

                    for frame, metrics in frame_generator:
                        if frame is not None:
                            with video_placeholder.container():
                                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            frame_count += 1
                            if total_frames > 0:
                                progress = frame_count / total_frames
                                progress_bar.progress(progress, text=f"Analyzing... {int(progress*100)}%")
                        if metrics:
                            followup_metrics = metrics

                    progress_bar.progress(1.0, text="Analysis Complete!")

                    if followup_metrics:
                        st.session_state.followup_metrics = followup_metrics

                        # Run improvement analysis
                        analysis = improvement_engine.analyze_improvement(
                            st.session_state.original_metrics,
                            followup_metrics
                        )
                        feedback = improvement_engine.generate_improvement_feedback(analysis)
                        st.session_state.improvement_analysis = {
                            'analysis': analysis,
                            'feedback': feedback
                        }

                        st.success("Analysis complete! Let's see your improvement...")
                        st.session_state.page = 'improvement_results'
                        st.rerun()

            with col2:
                if st.button("← Cancel"):
                    st.session_state.page = 'advice'
                    st.rerun()

# ==================== IMPROVEMENT RESULTS PAGE ====================
elif st.session_state.page == 'improvement_results':
    if st.session_state.improvement_analysis is None:
        st.warning("No improvement analysis found.")
        if st.button("← Go to Analysis"):
            st.session_state.page = 'analysis'
            st.rerun()
    else:
        analysis = st.session_state.improvement_analysis['analysis']
        feedback = st.session_state.improvement_analysis['feedback']

        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        with nav_col1:
            if st.button("← Back to Original Results"):
                st.session_state.page = 'advice'
                st.rerun()
        with nav_col3:
            if st.button("🏠 Start New Analysis"):
                # Clear all analysis data
                st.session_state.final_metrics = None
                st.session_state.original_metrics = None
                st.session_state.original_advice = None
                st.session_state.followup_metrics = None
                st.session_state.improvement_analysis = None
                st.session_state.page = 'analysis'
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Your Improvement Analysis</p>', unsafe_allow_html=True)

        # Verdict Card
        verdict = analysis['overall_verdict']
        verdict_class = 'verdict-improved' if verdict in ['IMPROVED', 'SLIGHT_IMPROVEMENT'] else \
                       'verdict-regressed' if verdict in ['REGRESSED', 'SLIGHT_REGRESSION'] else 'verdict-maintained'

        verdict_icon = '🎉' if verdict in ['IMPROVED', 'SLIGHT_IMPROVEMENT'] else \
                      '📉' if verdict in ['REGRESSED', 'SLIGHT_REGRESSION'] else '➡️'

        st.markdown(f"""
        <div class="{verdict_class}">
            <h2 style="margin-top: 0; color: white;">{verdict_icon} {verdict.replace('_', ' ')}</h2>
            <p style="margin-bottom: 0; font-size: 1.1rem; color: white;">{analysis['verdict_description']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Accuracy Comparison
        st.markdown("### Accuracy Score Comparison")

        acc_col1, acc_col2, acc_col3 = st.columns([1, 0.5, 1])

        with acc_col1:
            st.markdown(f"""
            <div class="accuracy-box original">
                <p style="font-size: 0.9rem; color: #86868b; margin-bottom: 0.5rem;">ORIGINAL</p>
                <p class="accuracy-value">{analysis['original_accuracy']:.0f}</p>
                <p style="margin-bottom: 0; color: #86868b;">/100</p>
            </div>
            """, unsafe_allow_html=True)

        with acc_col2:
            # Show change indicator
            change = analysis['accuracy_change']
            if change > 0:
                badge_class = 'improvement-badge-positive'
                change_text = f'+{change:.0f}'
            elif change < 0:
                badge_class = 'improvement-badge-negative'
                change_text = f'{change:.0f}'
            else:
                badge_class = 'improvement-badge-neutral'
                change_text = '0'

            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; padding-top: 2rem;">
                <div class="{badge_class}">{change_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with acc_col3:
            st.markdown(f"""
            <div class="accuracy-box followup">
                <p style="font-size: 0.9rem; color: #0071e3; margin-bottom: 0.5rem;">FOLLOW-UP</p>
                <p class="accuracy-value">{analysis['followup_accuracy']:.0f}</p>
                <p style="margin-bottom: 0; color: #86868b;">/100</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Two column layout for detailed comparison
        detail_col1, detail_col2 = st.columns([1, 1], gap="large")

        with detail_col1:
            st.markdown("### Metric-by-Metric Comparison")

            # Side-by-side metric comparison
            metrics_to_show = [
                ('max_elbow_angle', 'Elbow Extension', '°'),
                ('weight_transfer_amount', 'Weight Transfer', '%'),
                ('body_rotation_total', 'Body Rotation', '°'),
                ('head_movement', 'Head Stability', '°')
            ]

            for metric_key, metric_name, unit in metrics_to_show:
                if metric_key in analysis['metric_comparisons']:
                    comp = analysis['metric_comparisons'][metric_key]
                    orig_val = comp['original_value']
                    follow_val = comp['followup_value']
                    status = comp['status']

                    # Determine card class
                    card_class = 'metric-improved' if status == 'improved' else \
                               'metric-regressed' if status == 'regressed' else 'metric-maintained'

                    # Status indicator
                    status_icon = '↑' if status == 'improved' else '↓' if status == 'regressed' else '→'
                    status_color = '#30d158' if status == 'improved' else '#ff453a' if status == 'regressed' else '#86868b'

                    st.markdown(f"""
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>{metric_name}</strong>
                            <span style="color: {status_color}; font-weight: 600;">{status_icon}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                            <span>{orig_val:.1f}{unit} → {follow_val:.1f}{unit}</span>
                            <span style="color: {status_color};">{comp['score_change']:+.0f} pts</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Chart comparison
            st.markdown("### Body Rotation Comparison")

            original_rotation = st.session_state.original_metrics.get('body_rotation_over_time', [])
            followup_rotation = st.session_state.followup_metrics.get('body_rotation_over_time', [])

            if original_rotation and followup_rotation:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=list(range(len(original_rotation))),
                    y=original_rotation,
                    mode='lines',
                    name='Original',
                    line=dict(color='#86868b', width=2, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=list(range(len(followup_rotation))),
                    y=followup_rotation,
                    mode='lines',
                    name='Follow-up',
                    line=dict(color='#0071e3', width=3)
                ))

                fig.update_layout(
                    xaxis_title='Frame',
                    yaxis_title='Rotation (°)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    xaxis=dict(gridcolor='#e0e0e0', showgrid=True),
                    yaxis=dict(gridcolor='#e0e0e0', showgrid=True)
                )

                st.plotly_chart(fig, use_container_width=True)

        with detail_col2:
            st.markdown("### Coach's Feedback")

            # Summary
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;">
                <p style="margin: 0; line-height: 1.6;">{feedback['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Areas of improvement
            if feedback['improvements']:
                st.markdown("#### Areas You Improved")
                for imp in feedback['improvements']:
                    st.markdown(f'<div class="metric-improved">{imp}</div>', unsafe_allow_html=True)

            # Areas that regressed (HONEST FEEDBACK)
            if feedback['regressions']:
                st.markdown("#### Areas Needing Attention")
                for reg in feedback['regressions']:
                    st.markdown(f'<div class="metric-regressed">{reg}</div>', unsafe_allow_html=True)

            # Focus areas
            if feedback['focus_areas']:
                st.markdown("#### Focus Areas for Next Practice")
                for focus in feedback['focus_areas']:
                    st.markdown(f'<div class="focus-area-card">{focus}</div>', unsafe_allow_html=True)

            # Drills
            if feedback['drills']:
                st.markdown("#### Recommended Drills")
                for drill in feedback['drills']:
                    st.markdown(f'<div class="drill-card">🏋️ {drill}</div>', unsafe_allow_html=True)

        # Export options
        st.markdown("---")
        st.markdown("### Export Your Progress Report")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            improvement_summary = improvement_engine.get_improvement_summary(analysis, feedback)
            st.download_button(
                label="📥 Download Improvement Report",
                data=improvement_summary,
                file_name="cricket_improvement_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        with export_col2:
            if st.button("🔄 Try Another Follow-up", use_container_width=True):
                st.session_state.followup_metrics = None
                st.session_state.improvement_analysis = None
                st.session_state.page = 'track_improvement'
                st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.6;">
    <p>CricketSense AI Coach • Professional Biomechanical Analysis</p>
</div>
""", unsafe_allow_html=True)
