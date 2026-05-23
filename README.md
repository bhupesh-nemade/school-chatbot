# School Chatbot

This project provides a document-based school chatbot using LangChain, Pinecone, and OpenRouter. The original terminal chat is still available, and `app.py` adds a Streamlit UI.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file with:

```text
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key
HF_TOKEN=your_huggingface_token
```

## Run Streamlit UI

```powershell
streamlit run app.py
```

## Run Terminal Chat

```powershell
python -m chatbot.terminal_chat
```

## Note

If Python fails with a path like `C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe`, delete and recreate the `venv` folder. Virtual environments store absolute interpreter paths and are not portable between machines.
