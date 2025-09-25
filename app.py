import streamlit as st
import os
import tempfile
import subprocess
import cv2
import speech_recognition as sr
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Setup Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')
search_tool = DuckDuckGoSearchRun()

st.set_page_config(page_title="Agentic AI Multimodal Analyzer", layout="wide")
st.title("Agentic AI Multimodal Analyzer")
st.markdown("AI agent capable of deciding actions and analyzing multimedia using Gemini API")

# Session state for memory
if "past_analyses" not in st.session_state:
    st.session_state.past_analyses = []

# Helper functions
def save_temp_file(uploaded_file):
    if uploaded_file:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    return None

def transcribe_audio(video_path):
    audio_path = video_path + "_audio.wav"
    subprocess.run(
        ['ffmpeg', '-i', video_path, '-ac', '1', '-ar', '16000', audio_path, '-y'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    recognizer = sr.Recognizer()
    transcription = ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)
    except Exception as e:
        transcription = f"Transcribe Failed: {e}"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return transcription

# Sidebar Navigation
st.sidebar.title("Navigate")
page = st.sidebar.radio("Go to", ["Agentic Analysis", "Past Analyses"])

# Pages
if page == "Agentic Analysis":
    input_type = st.radio("Input Type:", ["Text Query", "Image", "Video"])
    uploaded_file = None
    query = ""

    if input_type == "Text Query":
        query = st.text_input("Enter your query")
    elif input_type in ["Image", "Video"]:
        uploaded_file = st.file_uploader(f"Upload {input_type}", type=["png","jpg","jpeg"] if input_type=="Image" else ["mp4","mov","avi"])

    if st.button("Analyze"):
        output = agent(query, uploaded_file, input_type.lower())
        st.markdown("### Analysis Result:")
        st.write(output)
        # Save to memory
        st.session_state.past_analyses.append({
            "type": input_type,
            "file": uploaded_file.name if uploaded_file else None,
            "query": query,
            "result": output
        })

elif page == "Past Analyses":
    st.header("Past Analyses")
    if st.session_state.past_analyses:
        for i, analysis in enumerate(st.session_state.past_analyses):
            st.subheader(f"{i+1}. {analysis['type']}")
            if analysis.get("file") and input_type != "Text Query":
                if analysis["type"] == "Image":
                    st.image(save_temp_file(uploaded_file), width=200)
                elif analysis["type"] == "Video":
                    st.video(save_temp_file(uploaded_file))
            if analysis.get("query"):
                st.markdown(f"*Query:* {analysis['query']}")
            st.markdown(f"*Result:* {analysis['result']}")
    else:
        st.info("No past analyses yet.")