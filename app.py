import streamlit as st
from moviepy import VideoFileClip
import whisper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import tempfile
import os
from pathlib import Path

# --- Constants ---
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
MODEL_NAME = "facebook/bart-large-cnn"  # Using a summarization model
WHISPER_MODEL_DIR = os.path.join(HF_CACHE, "whisper")
Path(WHISPER_MODEL_DIR).mkdir(parents=True, exist_ok=True)

# --- Load Models ---
@st.cache_resource
def load_whisper():
    try:
        return whisper.load_model("base", download_root=WHISPER_MODEL_DIR)
    except Exception as e:
        st.error(f"Whisper load failed: {e}")
        return None

@st.cache_resource
def load_summarization_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

# --- Processing Functions ---
def extract_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video = VideoFileClip(tmp_file.name)
        audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
        video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe(audio_path):
    model = load_whisper()
    return model.transcribe(audio_path)["text"] if model else "Whisper model not available"

def summarize(text):
    tokenizer, model, device = load_summarization_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.title("🎥 Video-to-Text with Fast Summarization")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
if uploaded_file:
    st.video(uploaded_file)
    if st.button("Process Video"):
        with st.spinner("Extracting audio..."):
            audio_file = extract_audio(uploaded_file)
            st.audio(audio_file)
        with st.spinner("Transcribing..."):
            transcript = transcribe(audio_file)
            st.subheader("Transcript:")
            st.write(transcript)
        with st.spinner("Summarizing..."):
            st.subheader("Summary:")
            st.write(summarize(transcript))