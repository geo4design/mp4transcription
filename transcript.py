import streamlit as st
import os
import tempfile
from moviepy.editor import VideoFileClip
import torch
from faster_whisper import WhisperModel
import numpy as np
from pydub import AudioSegment
import time
import threading
from datetime import timedelta

# Global variables
stop_transcription = threading.Event()
transcript_lock = threading.Lock()
full_transcript = ""
formatted_transcript = ""

# Cache the whisper model loading
@st.cache_resource
def load_whisper_model(model_name):
    """Load and cache the Whisper model"""
    return WhisperModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "float32")

def extract_audio(video_path, audio_path, progress_bar):
    """Extract audio from video file"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, logger=None)
    video.close()
    progress_bar.progress(0.2, "Audio extracted successfully")

def process_audio(audio_path):
    """Convert audio to format compatible with Whisper"""
    # Load audio and normalize
    audio = AudioSegment.from_file(audio_path)
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    # Export as WAV with 16kHz sample rate (optimal for Whisper)
    optimized_path = audio_path.replace(".wav", "_optimized.wav")
    audio.export(optimized_path, format="wav", parameters=["-ar", "16000"])
    return optimized_path

def transcribe_audio_with_whisper(audio_path, model, progress_bar, info, transcript_output):
    """Transcribe audio using Whisper model with timestamps"""
    global full_transcript, formatted_transcript
    
    # Process audio to optimize for Whisper
    optimized_audio_path = process_audio(audio_path)
    
    # Update progress
    progress_bar.progress(0.3, "Audio optimized for transcription")
    
    # Load audio
    info.text("Starting transcription...")
    
    # Determine if CUDA (GPU) is available
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    info.text(f"Transcribing using Whisper on {device}...")
    
    # Transcribe with timestamps
    segments, info_dict = model.transcribe(
        optimized_audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,  # Filter out non-speech
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Update progress
    progress_bar.progress(0.8, "Processing transcription results")
    
    # Format transcripts
    full_transcript = ""
    formatted_transcript = ""
    
    # Process segments
    for segment in segments:
        # Add to plain transcript
        full_transcript += segment.text + " "
        
        # Add to formatted transcript with timestamps
        start_time = str(timedelta(seconds=round(segment.start)))
        formatted_transcript += f"[{start_time}] {segment.text.strip()}\n"
    
    # Clean up transcript
    full_transcript = full_transcript.strip()
    
    # Update the transcript in the UI
    transcript_output.text_area("Transcript:", full_transcript, height=300)
    
    # Complete progress
    progress_bar.progress(1.0, "Transcription completed")
    
    return full_transcript, formatted_transcript

def main():
    st.title("Video Transcription with Whisper")
    
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'formatted_transcript' not in st.session_state:
        st.session_state.formatted_transcript = ""
    
    # Model selection
    model_size = st.sidebar.selectbox(
        "Select Whisper Model Size", 
        ["tiny", "base", "small", "medium", "large-v2"],
        index=2,  # Default to "small"
        help="Larger models are more accurate but require more processing power and memory"
    )
    
    st.sidebar.info(
        "Model sizes:\n"
        "- tiny: Fast but less accurate (1GB RAM)\n"
        "- base: Good balance for speed (1GB RAM)\n"
        "- small: Better accuracy (2GB RAM)\n"
        "- medium: High accuracy (5GB RAM)\n"
        "- large-v2: Best accuracy (10GB RAM)"
    )
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        st.sidebar.success("GPU acceleration available! Transcription will be faster.")
    else:
        st.sidebar.warning("GPU not available. Using CPU for processing (slower).")
    
    # Language settings
    auto_detect = st.sidebar.checkbox("Auto-detect language", value=True)
    if not auto_detect:
        language = st.sidebar.selectbox(
            "Select language", 
            ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja"],
            index=0
        )
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            video_path = os.path.join(temp_dir, "temp_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            
            audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            
            # UI elements
            progress_bar = st.progress(0)
            info = st.empty()
            transcript_output = st.empty()
            
            col1, col2, col3 = st.columns(3)
            transcribe_button = col1.button("Transcribe")
            stop_button = col2.button("Stop Transcription")
            
            if transcribe_button:
                stop_transcription.clear()
                
                try:
                    # Load Whisper model
                    with st.spinner(f"Loading Whisper {model_size} model..."):
                        model = load_whisper_model(model_size)
                    
                    # Extract audio
                    with st.spinner("Extracting audio..."):
                        extract_audio(video_path, audio_path, progress_bar)
                    
                    # Transcribe
                    info.text("Starting transcription...")
                    st.session_state.transcript, st.session_state.formatted_transcript = transcribe_audio_with_whisper(
                        audio_path, model, progress_bar, info, transcript_output
                    )
                    
                    if not stop_transcription.is_set():
                        st.success("Transcription completed!")
                    else:
                        st.warning("Transcription was stopped by the user.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    info.empty()
            
            if stop_button:
                stop_transcription.set()
                st.warning("Stopping transcription... Please wait.")

            # Display tabs for different transcript formats
            if st.session_state.transcript:
                tab1, tab2 = st.tabs(["Simple Transcript", "Transcript with Timestamps"])
                
                with tab1:
                    st.text_area("Transcript:", st.session_state.transcript, height=300)
                    if st.button("Copy Simple Transcript"):
                        st.code(st.session_state.transcript)
                        st.success("Simple transcript copied to clipboard!")
                
                with tab2:
                    st.text_area("Transcript with Timestamps:", st.session_state.formatted_transcript, height=300)
                    if st.button("Copy Timestamped Transcript"):
                        st.code(st.session_state.formatted_transcript)
                        st.success("Timestamped transcript copied to clipboard!")
            
                # Save transcript options
                save_format = st.radio("Save transcript format:", ["Simple", "With Timestamps", "Both (separate files)"])
                
                if st.button("Save Transcript"):
                    try:
                        # Default save path
                        save_dir = os.path.expanduser("~/Downloads")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        if save_format == "Simple" or save_format == "Both (separate files)":
                            simple_path = os.path.join(save_dir, "transcript_simple.txt")
                            with open(simple_path, "w", encoding="utf-8") as f:
                                f.write(st.session_state.transcript)
                                
                        if save_format == "With Timestamps" or save_format == "Both (separate files)":
                            ts_path = os.path.join(save_dir, "transcript_with_timestamps.txt")
                            with open(ts_path, "w", encoding="utf-8") as f:
                                f.write(st.session_state.formatted_transcript)
                        
                        st.success(f"Transcript saved to Downloads folder!")
                    except Exception as e:
                        st.error(f"Error saving transcript: {str(e)}")

if __name__ == "__main__":
    main()
