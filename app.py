import html
import os

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# The fine-tuned model lives on the Hugging Face Hub, not in this repo -
# that keeps GitHub small and lets the app always pull the latest weights.
# Replace the default below with your own repo id after you push the
# model (see README.md), or set the MODEL_REPO_ID environment variable /
# Streamlit secret instead of editing the code.
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "jussyanosama/opus-mt-en-ar-finetuned")

NUM_BEAMS = 5           # same beam width used for evaluation in the notebook
MAX_INPUT_LENGTH = 128  # same max length used for training/evaluation


# ----------------------------------------------------------------------
# Model loading - cached so this only runs once per app session, not on
# every button click.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the translation model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO_ID).to(device)
    model.eval()
    return tokenizer, model, device


def translate(text, tokenizer, model, device):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH
    ).to(device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, max_length=MAX_INPUT_LENGTH, num_beams=NUM_BEAMS
        )
    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="English to Arabic Translator", page_icon="🌐", layout="centered")

# Arabic reads right-to-left, so the output box gets its own small style
# rather than relying on the browser to guess the direction.
st.markdown(
    """
    <style>
    .arabic-output {
        direction: rtl;
        text-align: right;
        font-size: 1.25rem;
        line-height: 1.9;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        min-height: 3rem;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("English \u2192 Arabic Translator")
st.caption("Fine-tuned OPUS-MT (Helsinki-NLP/opus-mt-en-ar) \u2014 demo")

tokenizer, model, device = load_model()

english_text = st.text_area(
    "English text", height=150, placeholder="Type or paste English text here..."
)

if st.button("Translate", type="primary"):
    if english_text.strip() == "":
        st.warning("Please enter some English text first.")
    else:
        with st.spinner("Translating..."):
            arabic_text = translate(english_text, tokenizer, model, device)
        st.subheader("Arabic translation")
        st.markdown(
            f'<div class="arabic-output">{html.escape(arabic_text)}</div>',
            unsafe_allow_html=True,
        )
