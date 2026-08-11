# English → Arabic Translator (Streamlit Demo)

A small Streamlit app that serves a fine-tuned [OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar)
English → Arabic translation model. The model was fine-tuned on a cleaned
~137k-pair corpus built from UN Parallel Corpus, News Commentary, and
TED2020 (see the accompanying notebook for the full pipeline).

## How the model is hosted

The fine-tuned model is **not** stored in this repository. It's about
300MB, which is well past what belongs in a git history (and over
GitHub's 100MB hard file limit). Instead, the model lives on the
**Hugging Face Hub**, and `app.py` downloads it at startup with
`AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO_ID)` — the same API
used throughout the training notebook. This keeps the GitHub repo tiny,
makes the model easy to version/update independently of the app code,
and needs no extra infrastructure.

### One-time step: push your fine-tuned model to the Hub

Run this once, from Colab (right after training, or pointing at your
saved Google Drive copy) or from anywhere with the model files on disk:

```python
from huggingface_hub import login
login()  # paste a Hugging Face access token with "write" permission

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "/content/drive/MyDrive/EN_AR_Machine_Translation/best_model"  # or your local path
REPO_ID = "your-username/opus-mt-en-ar-finetuned"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)
```

Get a token from https://huggingface.co/settings/tokens (scope: **write**).
The repo is created automatically the first time you push to it, and is
**public by default** — this app is written assuming that (no auth
needed to download it). If you'd rather keep the model private, see
"Using a private model repo" below.

Once pushed, update `MODEL_REPO_ID` in `app.py` (or set it as an
environment variable / Streamlit secret — see below) to your repo id.

## Project files

- `app.py` — the Streamlit app (single file, no build step)
- `requirements.txt` — Python dependencies
- `.gitignore` — keeps virtual envs, caches, and any local model copies out of git
- `README.md` — this file

## Running locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# either edit MODEL_REPO_ID directly in app.py, or:
export MODEL_REPO_ID="your-username/opus-mt-en-ar-finetuned"

streamlit run app.py
```

The first run will download the model from the Hub (a few hundred MB,
cached afterward in `~/.cache/huggingface`), so it may take a minute
before the app is responsive. After that, translations run on CPU by
default and are fast for single sentences (this is a small ~77M
parameter model, not NLLB).

## Deploying (Streamlit Community Cloud)

1. Push this folder to a new GitHub repository (the model itself is
   **not** included, per above — just these four files).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   click "New app."
3. Point it at your repo and `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   MODEL_REPO_ID = "your-username/opus-mt-en-ar-finetuned"
   ```
   (Optional if you already hardcoded it in `app.py` — the environment
   variable just makes it easy to change later without a new commit.)
5. Deploy. The first load will be slow (downloading the model); after
   that it's cached for the life of the container.

Any other host that runs a plain Python app (Hugging Face Spaces with a
Streamlit SDK, Render, Railway, a VM, etc.) works the same way — install
`requirements.txt` and run `streamlit run app.py`. Hugging Face Spaces in
particular is a natural fit here since your model already lives on the
Hub.

## Using a private model repo

If you push the model as private instead of public, the app needs a
read token to download it. Add one line to `app.py`'s `load_model()`:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, token=st.secrets["HF_TOKEN"])
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO_ID, token=st.secrets["HF_TOKEN"]).to(device)
```

and add `HF_TOKEN` (a token with **read** access) to your Streamlit
secrets the same way as `MODEL_REPO_ID` above. For a graduation-project
demo, keeping the model public is simpler and avoids managing a secret
at all — recommended unless you have a specific reason not to.
