# School Chatbot

This project provides a document-based school chatbot using LangChain, Pinecone, and OpenRouter. The original terminal chat is still available, `app.py` adds a Streamlit UI, and `api.py` exposes the same RAG pipeline through FastAPI.

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

## Run FastAPI Backend

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Send a chat request:

```powershell
curl -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"What is the admission process?\"}"
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "FAQ_Handbook.pdf",
      "page": "3",
      "preview": "..."
    }
  ]
}
```

## Run Terminal Chat

```powershell
python -m chatbot.terminal_chat
```

## Note

If Python fails with a path like `C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe`, delete and recreate the `venv` folder. Virtual environments store absolute interpreter paths and are not portable between machines.
