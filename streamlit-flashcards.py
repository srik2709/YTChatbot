import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import whisper
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

st.set_page_config(page_title="YouTube ➜ Summary & Flashcards", page_icon="🎓", layout="wide")

# ------------------------------------------------------------------
# CACHED MODELS (only loaded once per session)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("base", device="cpu")


@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline(
        "summarization", model="facebook/bart-large-cnn", device=-1  # CPU only
    )


@st.cache_resource(show_spinner=False)
def load_qg():
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
    model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl")
    return tokenizer, model


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def download_audio(url: str, workdir: Path) -> Path:
    """Download highest‑quality audio from a YouTube URL to MP3."""
    st.info("Downloading audio… this can take a minute ⏳")
    audio_path = workdir / "%\(id)s.%(ext)s"  # yt‑dlp pattern
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "-o",
        str(audio_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        st.error("yt-dlp failed. Details in console log.")
        st.text(result.stderr.decode())
        raise RuntimeError("Audio download failed")
    # yt‑dlp expands pattern → first file in workdir
    return next(workdir.glob("*.mp3"))


def split_into_chunks(text: str, max_words: int = 450) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def summarize_chunks(chunks: List[str], summarizer) -> str:
    mini_summaries = []
    for chunk in chunks:
        if len(chunk.split()) < 30:
            continue
        summary = summarizer(
            chunk,
            max_length=300,
            min_length=100,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
        mini_summaries.append(summary)
    if not mini_summaries:
        return "⚠️ Could not summarize any chunks."  # early exit
    combined = " ".join(mini_summaries)
    final = summarizer(
        combined, max_length=500, min_length=150, do_sample=False, truncation=True
    )[0]["summary_text"]
    return final


def generate_flashcards(summary: str, tokenizer, model) -> List[Tuple[str, str]]:
    import torch

    device = torch.device("cpu")
    flashcards = []
    sentences = summary.split(". ")
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 7:
            continue
        prompt = f"generate question: {sent}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(
            input_ids, max_length=64, num_beams=4, early_stopping=True
        )
        question = tokenizer.decode(output[0], skip_special_tokens=True)
        flashcards.append((question.strip(), sent))
    return flashcards


# ------------------------------------------------------------------
# UI LAYOUT
# ------------------------------------------------------------------

st.title("🎬 YouTube → 📜 Summary → 🃏 Flashcards")
st.markdown(
    "Convert any YouTube lecture or talk into a concise summary **and** quiz‑style flashcards, all in your browser.")

url = st.text_input("Paste a YouTube URL to begin", placeholder="https://www.youtube.com/watch?v=…")

col1, col2 = st.columns([1, 4])
with col1:
    start_btn = st.button("Generate", type="primary")
with col2:
    sample = st.button("Try sample video")

if sample:
    url = "https://www.youtube.com/watch?v=LPZh9BOjkQs"
    st.experimental_rerun()

if start_btn and url:
    whisper_model = load_whisper_model()
    summarizer = load_summarizer()
    tokenizer, qg_model = load_qg()

    with st.spinner("Processing… grab a coffee ☕"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mp3_path = download_audio(url, tmpdir)

            st.success("Audio downloaded ✔️")
            st.info("Transcribing with Whisper…")
            transcript = whisper_model.transcribe(str(mp3_path))
            text = transcript["text"]
            st.success("Transcript ready ✔️")

        with st.expander("📄 Raw transcript (first 1500 chars)"):
            st.write(text[:1500] + "…")

        st.info("Summarising…")
        chunks = split_into_chunks(text)
        summary = summarize_chunks(chunks, summarizer)
        st.success("Summary completed ✔️")

        st.subheader("📝 Video Summary")
        st.markdown(summary)

        st.info("Generating flashcards…")
        flashcards = generate_flashcards(summary, tokenizer, qg_model)
        st.success(f"Created {len(flashcards)} flashcards ✔️")

        st.subheader("🎯 Flashcards (Q → A)")
        for i, (q, a) in enumerate(flashcards, 1):
            with st.expander(f"{i}. {q}"):
                st.write(a)

        # Optional download
        if flashcards:
            csv_lines = ["Question,Answer"] + [f'"{q}","{a}"' for q, a in flashcards]
            csv_data = "\n".join(csv_lines).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV", data=csv_data, file_name="flashcards.csv", mime="text/csv"
            )

else:
    st.caption("👈 Enter a valid YouTube link and click *Generate*. Processing happens entirely on CPU, so please be patient.")