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

def analyze_content(prompt, image_path=None, transcription=None):
    content = prompt
    if transcription:
        content += f"\n\nTranscription: {transcription}"

    try:
        if image_path:
            with open(image_path, "rb") as img:
                response = model.generate_content(
                    [content, {"mime_type": "image/jpeg", "data": img.read()}]
                )
        else:
            response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Analysis Failed: {e}"

def analyze_video_frames(video_path, transcription):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps  # 1 frame per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    analyses = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:  # sample every second
            temp_frame = f"{video_path}frame{i}.jpg"
            cv2.imwrite(temp_frame, frame)
            result = analyze_content("Analyze this frame of the video.", image_path=temp_frame)
            analyses.append(result)
            os.remove(temp_frame)
        i += 1
    cap.release()

    final_summary = analyze_content(
        f"Summarize the following frame analyses into a cohesive video analysis:\n\n{analyses}",
        transcription=transcription
    )
    return final_summary, duration, len(analyses)

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