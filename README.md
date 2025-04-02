# 🎥 Video-to-Text with Fast Summarization using OpenAI Whisper, Facebook's BART-Large-CNN model, Streamlit and Python Code

This app allows you to upload a video file, extract its audio, transcribe the speech into text using OpenAI Whisper, and summarize the transcription using Facebook's BART-Large-CNN model. It uses Streamlit for the user interface and Python code to handle the video processing and model operations.

# Features
- Upload video files (MP4, MOV, AVI).
- Extracts audio from the uploaded video.
- Transcribe Speech: The app uses OpenAI's Whisper model to transcribe speech from the uploaded video.
- Summarize Text: The app uses Facebook's BART-Large-CNN model to generate concise summaries from the transcribed text.
- The app will automatically download the required models if they are not found locally, saving you the hassle of manually downloading them.

# Models Used
- Whisper (Base) - A powerful speech recognition model developed by OpenAI for accurate transcription of spoken words.
- Facebook BART-Large-CNN - A transformer-based text summarization model that efficiently shortens long transcripts into concise summaries.

Both models are loaded locally using Hugging Face caching to avoid unnecessary downloads, but the app will automatically download them if they are missing from the cache.

# Libraries Used
Here are the main libraries used in this project:

- Streamlit - Framework for building interactive web applications in Python.
- MoviePy - Library for video editing and audio extraction.
- Whisper - OpenAI’s speech recognition model for transcribing audio to text.
- Transformers - Hugging Face library for pre-trained transformer models such as BART-Large-CNN.
- Torch - PyTorch, a deep learning library used for model inference and device management.

# Installation
Clone the Repository
git clone https://github.com/yourusername/video-to-text-summarizer.git  
cd video-to-text-summarizer 
 
# Install Dependencies
To install all the required libraries for this project, use the following command:

pip install -r requirements.txt  
Here is the content for the requirements.txt file:
streamlit
moviepy
whisper
transformers
torch

# Install CUDA-enabled PyTorch (Recommended for NVIDIA GPU)
If you have an NVIDIA GPU, install PyTorch with CUDA support for faster inference:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
(Ensure the CUDA version matches your system’s configuration.)

# Run the Application
streamlit run app.py  

# How to Use the App
Upload a Video:

- Open the app in your browser after running streamlit run app.py.
- You'll be presented with a file uploader. Click on the "Upload video" button to select and upload a video file (supports MP4, MOV, and AVI formats).

Process the Video:

- Once the video is uploaded, the app will automatically extract the audio and display a preview of the video.
- After the video is uploaded, click the "Process Video" button to begin processing. The app will first extract audio from the video, then transcribe the speech into text using Whisper, and finally summarize the transcribed text using the BART-Large-CNN model.

View the Results:

- After processing, you will see the transcribed text displayed under the "Transcript" section.
- Below that, you'll find the summarized version of the transcript under the "Summary" section.

Download Audio:

- You can listen to the extracted audio by clicking the audio player below the uploaded video.
Re-process or Upload New Video:
- To process another video, simply upload a new one by clicking the "Upload video" button again and repeat the steps.

# How the Summarizing Works
1. Once the app transcribes the speech from the video, the next step is summarization. Here's how it works:
2. The transcribed text is fed into the BART-Large-CNN model, a type of machine learning model specifically designed for understanding and summarizing long pieces of text.
3. The model looks at the whole transcript and identifies the key points and important ideas.
4. It then shortens the transcript into a concise summary that captures the main messages while leaving out unnecessary details.
5. The result is a brief, easy-to-read summary of the video's content, making it quick to understand the core information.


