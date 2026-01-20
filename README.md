# LangExtract Local App

Small Streamlit app to run LangExtract locally with cloud API keys or Ollama.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API Key (optional)

Create a `.env` file with your key:

```bash
LANGEXTRACT_API_KEY=your-key-here
```

If you prefer OpenAI, set `OPENAI_API_KEY` and use a `gpt-*` model ID.

## Run

```bash
streamlit run app.py
```

## Notes

- For Ollama: install it, run `ollama serve`, then set a local model like
  `gemma2:2b` and enable "Use Ollama (local)" in the sidebar.
- OpenAI models typically require `fence_output=true` and
  `use_schema_constraints=false`.
- Uploads support `txt`, `md`, `json`, `csv`, `pdf`, and `docx`.
- PDF/DOCX uploads include basic table extraction and layout hints.
- For OCR on scanned PDFs, install Tesseract and Poppler, then enable
  "Use OCR for PDFs if no text is found".