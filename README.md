# Marginalia: chat with your documents

A single-page Streamlit dashboard for RAG (Retrieval-Augmented Generation). Upload PDF, TXT or DOCX files, ask questions, and get concise answers with the exact source passages underneath.

- **Pipeline:** upload → extract text → chunk → embed → FAISS index (in memory) → retrieve top passages → LLM answers from those passages only
- **Providers:** Google Gemini or OpenAI, chosen in code
- **Layout:** everything is on one dashboard (documents, stats, settings on the left, conversation on the right). There is no sidebar.

## Files

```
app.py                    the application
requirements.txt          dependencies
.streamlit/config.toml    forces the light theme so widgets never turn dark
.env                      optional, API keys (not committed)
```

Keep the folder structure: `.streamlit/config.toml` must sit next to `app.py`.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## API key: set it in code

Open `app.py` and edit the block near the top:

```python
GOOGLE_API_KEY = "paste-your-gemini-key-here"
OPENAI_API_KEY = ""
PROVIDER = "auto"        # "auto", "Google Gemini" or "OpenAI"
```

If a key is left empty, the app falls back to the same-named variable in your environment or a `.env` file:

```env
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
```

`PROVIDER = "auto"` uses whichever provider has a key (OpenAI first if both do). Never share or commit `app.py` once it contains a real key.

## Run

```powershell
streamlit run app.py
```

## Using the dashboard

1. Drop files into the **Documents** card and select **Process documents**.
2. Ask a question on the right, or click a starter question.
3. Open **Sources** under any answer to see the passages it used. Each file has its own colour, shared by its chip and its source cards.

**Settings** (left column) holds the model names, chunk size and overlap, and passages per question. Changing the embedding model, chunk size or overlap needs a fresh **Process documents**. **Clear chat** keeps the index; **Reset all** removes everything.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No API key found" banner | Paste the key into `app.py` (or `.env`), save; the app reloads. |
| "The provider rejected this API key" | Wrong key, or it belongs to the other provider. |
| "Rate-limiting ... out of quota" | Wait, or check billing. Free tiers have low per-minute limits. |
| "The chat model was not available for this key" | Google closed `gemini-2.5-flash` to new projects. The app now defaults to `gemini-3.8-flash` and automatically tries other current models; you can also edit the name under Settings. |
| "No extractable text" for a PDF | It is probably a scanned image; run OCR on it first. |
| Widgets look dark | The app forces light colours itself. For the most reliable result also keep `.streamlit/config.toml` next to `app.py` (start Streamlit from that folder), then restart. |

## Privacy

Text from your documents is sent to the chosen provider to create embeddings and answers. The FAISS index lives in memory only and is discarded when the session ends.
