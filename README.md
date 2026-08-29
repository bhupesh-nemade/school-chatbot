# School Chatbot

This project provides a document-based school chatbot using **LangChain, Pinecone, OpenRouter, Redis, RedisVL, FastAPI, and Streamlit**.

The chatbot uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions from school documents. A **Redis semantic cache** is used to avoid repeating Pinecone retrieval and LLM generation for previously answered, cache-eligible questions.

The original terminal chat is still available, `app.py` provides the Streamlit UI, and `api.py` exposes the same RAG pipeline through FastAPI.

## Architecture

```text
User Question
      │
      ▼
Cache Eligibility Check
      │
      ├── Not eligible
      │      └── RAG Pipeline
      │
      ▼
Redis Semantic Cache
      │
      ├── HIT ───────────────► Return cached response
      │                         Pinecone skipped
      │                         LLM skipped
      │
      └── MISS
             │
             ▼
       OpenRouter BGE-M3
       Embedding API
             │
             ▼
          Pinecone
       Vector Retrieval
             │
             ▼
        OpenRouter LLM
             │
             ▼
       Response Generated
             │
             ▼
        Store in Redis
```

### Embedding architecture

The application uses **BGE-M3 through the OpenRouter API** rather than loading the BGE-M3 model locally.

```text
Question
   │
   ▼
RedisVL EmbeddingsCache
   │
   ├── Embedding HIT
   │      └── Reuse existing embedding
   │
   └── Embedding MISS
          │
          ▼
     OpenRouter API
          │
          ▼
       BGE-M3
          │
          ▼
     1024-d vector
          │
          ▼
     Redis embedding cache
```

This avoids loading the large BGE-M3 model on the local machine.

## Features

* Document-based question answering
* Retrieval-Augmented Generation (RAG)
* Pinecone vector search
* OpenRouter LLM integration
* OpenRouter BGE-M3 embeddings
* 1024-dimensional embeddings
* Redis semantic response caching
* RedisVL embedding caching
* Shared cache across users for eligible generic questions
* Conversation-history-aware cache bypass
* FastAPI backend
* Streamlit UI
* Terminal chat interface
* Source and page information in responses
* Cache HIT/MISS logging
* Configurable cache TTL and similarity threshold

## Requirements

* Python 3.11+
* Pinecone account/API key
* OpenRouter account/API key
* Redis server
* RedisVL
* Internet connection for OpenRouter and Pinecone

Redis can run locally through **WSL, Docker, or another Redis deployment**.

The Python application requires the Redis client and RedisVL packages:

```text
redis
redisvl
```

## Setup

### 1. Clone the project

```powershell
git clone <repository-url>
cd school-chatbot
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If you are using an existing project environment:

```powershell
.\chatbot_env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Redis

The default application configuration expects:

```text
REDIS_URL=redis://localhost:6379
```

If Redis is running through WSL, make sure Redis is running:

```bash
redis-cli PING
```

Expected:

```text
PONG
```

You can also verify the Redis server:

```bash
redis-cli INFO server
```

### 5. Configure environment variables

Create a `.env` file:

```text
PINECONE_API_KEY=your_pinecone_key

OPENROUTER_API_KEY=your_openrouter_key

HF_TOKEN=your_huggingface_token

REDIS_URL=redis://localhost:6379
```

`HF_TOKEN` is retained for Hugging Face-related components that may be used elsewhere in the project. BGE-M3 embeddings for the semantic cache are generated through OpenRouter.

**Do not commit `.env` to Git.**

## Redis Semantic Cache

The semantic cache stores previously generated responses in Redis.

For example:

```text
Question:
What documents are required for admission?
```

On the first request:

```text
CACHE MISS
      ↓
Pinecone
      ↓
LLM
      ↓
Response
      ↓
Redis STORE
```

On a subsequent eligible request:

```text
CACHE HIT
      ↓
Return cached response
```

Pinecone and the LLM are skipped during a cache hit.

### Cache eligibility

Generic questions without conversation history can use the shared semantic cache.

Questions that depend on conversation history are bypassed:

```text
history_turns > 0
        ↓
CACHE BYPASS
```

This prevents a previous conversation from incorrectly affecting a cached answer.

### Cache TTL

Cached responses use a configurable TTL.

When the TTL expires, Redis automatically removes the cached entry and the next eligible request goes through the normal RAG pipeline again.

## Verify Redis

Check whether Redis is reachable:

```powershell
python -c "import redis; from config import REDIS_URL; r=redis.from_url(REDIS_URL); print('PING:', r.ping()); print('DBSIZE:', r.dbsize())"
```

List semantic-cache entries:

```powershell
redis-cli --scan --pattern "school_chatbot_semantic_cache:*"
```

Inspect Redis keys:

```powershell
redis-cli KEYS "school_chatbot_semantic_cache:*"
```

For production environments, prefer `SCAN` rather than `KEYS` when inspecting large Redis databases.

## Run Streamlit UI

```powershell
streamlit run app.py
```

The application will provide a local Streamlit interface.

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

## Cache Testing

### Test a cache MISS

Use a new conversation and ask:

```text
What documents are required for admission?
```

Expected log:

```text
[CACHE] ELIGIBILITY | eligible=True
[CACHE] MISS
```

The request then continues through:

```text
Pinecone → LLM → Redis STORE
```

### Test a cache HIT

Ask the same question again from a **new conversation**:

```text
What documents are required for admission?
```

Expected:

```text
[CACHE] ELIGIBILITY | eligible=True
[CACHE] HIT | ... | Pinecone=SKIPPED | LLM=SKIPPED
```

The cached response should be displayed directly in the UI.

### Test cross-user caching

A second user can ask the same generic question from a new conversation.

Expected:

```text
eligible=True
CACHE HIT
Pinecone=SKIPPED
LLM=SKIPPED
```

This confirms that eligible generic responses are shared through Redis rather than being stored separately for each user.

## Redis Cache Key

Semantic response entries use keys similar to:

```text
school_chatbot_semantic_cache:<hash>
```

The hash identifies the cached entry.

Example:

```text
school_chatbot_semantic_cache:f46f70961374f76ba719b8f991fd8b2a81bdde4acf1ad6ed22d0a8cafe9ddcb9
```

## Performance

Without a cache hit, the request can involve:

```text
Embedding
   ↓
Pinecone retrieval
   ↓
LLM generation
```

With a cache hit:

```text
Embedding
   ↓
Redis semantic search
   ↓
Cached response
```

Therefore Pinecone retrieval and LLM generation are avoided for eligible cached questions.

The exact latency depends on Redis, network latency, OpenRouter embedding latency, and the application environment.

## Project Structure

```text
school-chatbot/
│
├── app.py
├── api.py
├── config.py
├── requirements.txt
├── README.md
│
├── chatbot/
│   ├── chain.py
│   ├── rag_service.py
│   ├── retriever.py
│   ├── semantic_cache.py
│   ├── terminal_chat.py
│   └── ...
│
├── documents/
│   └── ...
│
└── .env
```

## Troubleshooting

### Redis connection error

Check Redis:

```bash
redis-cli PING
```

Expected:

```text
PONG
```

Check the application configuration:

```powershell
python -c "from config import REDIS_URL; print(REDIS_URL)"
```

Expected:

```text
redis://localhost:6379
```

### Cache is always bypassed

Check the logs:

```text
[CACHE] ELIGIBILITY | eligible=False
```

If the source is:

```text
bypass_chat_history
```

the conversation contains previous turns. Test using a new conversation.

### Cache MISS when the question was previously answered

Verify that:

* Redis is running.
* The semantic-cache key still exists.
* The cache TTL has not expired.
* The question is cache-eligible.
* The embedding model configuration has not changed.
* The semantic distance threshold is appropriate.

### Python cannot import Redis

Make sure the project virtual environment is activated:

```powershell
.\venv\Scripts\Activate.ps1
```

Then install the dependencies:

```powershell
pip install -r requirements.txt
```

### Python virtual environment path error

If Python reports an error involving a path such as:

```text
C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe
```

delete and recreate the virtual environment:

```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Virtual environments store absolute interpreter paths and are not portable between machines.

## Security

* Never commit API keys.
* Keep `.env` out of version control.
* Use environment variables for secrets.
* Use authentication and authorization before exposing the FastAPI service publicly.
* Do not cache responses containing user-specific or sensitive information.
* Keep personalized/history-dependent questions outside the shared semantic cache.

## Development Notes

The application separates:

```text
LLM
Embedding Model
Vector Database
Semantic Response Cache
Embedding Cache
Conversation Memory
```

This allows each component to be optimized independently without changing the overall RAG architecture.
