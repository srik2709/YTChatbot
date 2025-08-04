import os
import streamlit as st
import whisper
import yt_dlp
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- App Configuration ---
st.set_page_config(
    page_title="Chat with YouTube",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_resource
def load_models():
    """Loads the Whisper and Sentence Transformer models."""
    whisper_model = whisper.load_model("base")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return whisper_model, embedding_model

def get_video_title(url):
    """Extracts the video title using yt-dlp."""
    try:
        ydl_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict.get('title', 'Untitled Video')
    except Exception:
        return "Untitled Video"

def download_and_transcribe(url, whisper_model):
    """Downloads audio and returns the full transcribed text."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    result = whisper_model.transcribe("downloaded_audio.mp3")
    os.remove("downloaded_audio.mp3") # Clean up the audio file
    return result['text']

def split_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Splits text into overlapping chunks of words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_vector_store(text_chunks, embedding_model):
    """Builds a FAISS vector store from text chunks."""
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
    
    # FAISS requires a 2D array of floats
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, text_chunks

def search_vector_store(query, embedding_model, index, texts, k=5):
    """Searches the vector store for the most relevant text chunks."""
    query_embedding = embedding_model.encode([query], convert_to_tensor=False).astype('float32')
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the original text chunks
    return [texts[i] for i in indices[0]]

# --- Streamlit UI ---

st.title("Chat with any YouTube Video")
st.markdown("Enter a YouTube URL, so that you can ask questions about the video's content.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get your free API key from groq.com")
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    process_button = st.button("Process Video", type="primary")
    st.markdown("---")
    st.info("This app uses a RAG pipeline with Groq's Llama3-8B model to chat with video content.")

# --- Main App Logic ---

# Initialize models
whisper_model, embedding_model = load_models()

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "video_title" not in st.session_state:
    st.session_state.video_title = ""

# --- Video Processing ---
if process_button and youtube_url:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Processing video... This may take a few minutes depending on the length of the video and your system's memory. The entire process will run in your system's memory. So please be patient."):
            try:
                # Reset state for new video
                st.session_state.messages = []
                st.session_state.vector_store = None
                
                st.session_state.video_title = get_video_title(youtube_url)
                st.info(f"Processing video: **{st.session_state.video_title}**")

                # 1. Download and Transcribe
                full_transcript = download_and_transcribe(youtube_url, whisper_model)
                
                # 2. Split text into chunks
                text_chunks = split_into_chunks(full_transcript)

                # 3. Build Vector Store
                index, chunks_for_store = build_vector_store(text_chunks, embedding_model)
                st.session_state.vector_store = (index, chunks_for_store)
                
                st.success("Video processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.vector_store = None # Ensure it's cleared on error

# --- Chat Interface ---
if st.session_state.video_title:
    st.header(f"Chat: *{st.session_state.video_title}*")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_query := st.chat_input("Ask a question about the video..."):
    if st.session_state.vector_store is None:
        st.error("Please process a video first.")
    elif not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar to chat.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve context and generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                index, texts = st.session_state.vector_store
                
                # Search for context
                context_chunks = search_vector_store(user_query, embedding_model, index, texts)
                context = "\n\n".join(context_chunks)

                # Display the context being used
                with st.expander("Show Context"):
                    st.write(context)

                # Prepare prompt for Groq API
                client = Groq(api_key=groq_api_key)
                prompt = f"""
                You are a helpful YouTube assistant. Your task is to answer the user's question, you can use the provided video transcript context"

                CONTEXT FROM THE VIDEO:
                ---
                {context}
                ---

                USER'S QUESTION: {user_query}
                """
                
                # Get and stream response from Groq
                try:
                    stream = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0,
                        stream=True,
                    )
                    
                    def stream_generator(stream):
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                yield chunk.choices[0].delta.content

                    response = st.write_stream(stream_generator(stream))
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"An error occurred with the Groq API: {e}")
