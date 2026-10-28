# Document Ingestion + Conversational RAG Backend

A FastAPI backend with two REST APIs:

1. **Document Ingestion API**: upload `.pdf` / `.txt` files, extract text, split it with one of two selectable chunking strategies, embed the chunks, store the vectors in **Qdrant**, and save document metadata in **PostgreSQL**.
2. **Conversational RAG API**: answer questions about the ingested documents with a **custom-built** RAG pipeline (no `RetrievalQAChain`, no LangChain chains), keep multi-turn chat memory in **Redis**, and book interviews (name, email, date, time) through the LLM, storing each booking in PostgreSQL.

No UI, no FAISS, no Chroma. The code is fully typed (`mypy --strict` and `ruff` both pass).

---

## Contents

- [Requirement checklist](#requirement-checklist)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Testing and quality checks](#testing-and-quality-checks)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## Requirement checklist

| Requirement | Where it lives |
|-------------|----------------|
| FastAPI REST APIs | `app/main.py`, `app/api/v1/endpoints/` |
| Upload `.pdf` / `.txt` | `app/api/v1/endpoints/documents.py`, `app/rag/loaders.py` |
| Two selectable chunking strategies | `app/rag/chunking/` (`fixed`, `recursive`) |
| Embeddings stored in a vector DB (not FAISS/Chroma) | `app/rag/embeddings.py`, `app/rag/vector_store.py` (Qdrant) |
| Metadata in a SQL DB | `app/models/`, `app/repositories/`, `alembic/` (PostgreSQL) |
| Custom RAG, no `RetrievalQAChain` | `app/rag/pipeline.py`, `retriever.py`, `query_rewriter.py`, `prompts.py` |
| Redis chat memory | `app/memory/redis_memory.py` |
| Multi-turn queries | History-aware query rewriting in `app/rag/query_rewriter.py` |
| Interview booking via the LLM | `app/services/booking_extractor.py`, `booking_service.py`, `booking_validation.py` |
| Booking info stored | `interview_bookings` table, `GET /api/v1/bookings` |
| Clean, modular, typed code | Layered structure, `mypy --strict`, `ruff` |

---

## Tech stack

| Concern | Choice |
|---------|--------|
| Web framework | FastAPI, Pydantic v2, pydantic-settings |
| Vector database | Qdrant (Qdrant Cloud free tier) |
| LLM and embeddings | Google Gemini (`google-genai` SDK), `gemini-embedding-001` for embeddings |
| Relational database | PostgreSQL, SQLAlchemy 2.0 (async, `asyncpg`), Alembic migrations |
| Chat memory | Redis (`redis-py` async) |
| PDF text extraction | `pypdf` |
| Tests and quality | pytest, pytest-asyncio, httpx, fakeredis, ruff, mypy (strict) |

Python 3.13 is required.

---

## Quick start

These commands are for Git Bash on Windows. On macOS or Linux, replace the activation line with `source .venv/bin/activate`.

### 1. Create the environment and install dependencies

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Prepare the external services

| Service | What you need |
|---------|---------------|
| **Gemini** | An API key from [Google AI Studio](https://aistudio.google.com) |
| **Qdrant** | A cluster URL and API key (a free cluster on [Qdrant Cloud](https://cloud.qdrant.io) works) |
| **PostgreSQL** | A running server and an empty database, for example `rag_db` |
| **Redis** | A Redis URL (a free database on Redis Cloud works) |

Create the PostgreSQL database:

```bash
psql -U postgres -h localhost -c "CREATE DATABASE rag_db;"
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in the values. See [Configuration](#configuration) for what each one means. Do not commit `.env`; it is already in `.gitignore`.

Before choosing models, you can list the ones your key can use:

```bash
python scripts/check_gemini.py
```

Then test the pair you picked:

```bash
python scripts/check_gemini.py <llm-model> <embedding-model>
```

The second command prints the real embedding dimension, which must match `EMBEDDING_DIMENSION` in `.env`.

### 4. Create the database tables

```bash
alembic upgrade head
```

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

On startup the app creates the Qdrant collection if it does not exist, and checks that Qdrant and Redis are reachable. If either is not, it refuses to start and says why.

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## Configuration

All settings are read from environment variables or `.env` and validated at startup (`app/core/config.py`). A missing required value stops the app with a clear error that names the field.

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `GEMINI_API_KEY` | yes | | Gemini API key |
| `GEMINI_LLM_MODEL` | yes | | Model used for intent detection, query rewriting and answers. Must support JSON output. |
| `GEMINI_EMBEDDING_MODEL` | yes | | Embedding model. The project was built and tested with `gemini-embedding-001`. |
| `EMBEDDING_DIMENSION` | yes | | Vector size produced by the embedding model (3072 for `gemini-embedding-001`) |
| `QDRANT_URL` | yes | | Qdrant URL including port, for example `https://xxxx.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | yes | | Qdrant API key |
| `QDRANT_COLLECTION` | no | `documents_chunks` | Collection name |
| `DATABASE_URL` | yes | | `postgresql+asyncpg://user:password@host:5432/dbname`. Special characters in the password must be percent-encoded. |
| `REDIS_URL` | yes | | `redis://default:password@host:port` |
| `CHAT_HISTORY_MAX_MESSAGES` | no | `20` | Messages kept per session (sliding window) |
| `CHAT_TTL_SECONDS` | no | `3600` | Idle time after which a session expires |
| `DEFAULT_CHUNK_SIZE` | no | `800` | Characters per chunk when the upload does not specify one |
| `DEFAULT_CHUNK_OVERLAP` | no | `100` | Overlap between chunks in characters |
| `RETRIEVAL_TOP_K` | no | `4` | Maximum chunks used as context |
| `RETRIEVAL_MIN_SCORE` | no | `0.6` | Minimum similarity for a chunk to count as relevant. Below it, the assistant refuses without calling the LLM. |
| `EMBEDDING_BATCH_SIZE` | no | `50` | Chunks per embedding request (the API limit is 100) |
| `EMBEDDING_MAX_RETRIES` | no | `5` | Retries for temporary embedding failures |
| `LLM_MAX_RETRIES` | no | `5` | Retries for temporary LLM failures |
| `LLM_TEMPERATURE` | no | model default | Sampling temperature for answers |
| `BOOKING_TIMEZONE` | no | `UTC` | IANA timezone that decides what "today" and "tomorrow" mean for bookings, for example `Asia/Kolkata` |
| `MAX_UPLOAD_MB` | no | `10` | Maximum upload size |
| `LOG_LEVEL` | no | `INFO` | Logging level |

`RETRIEVAL_MIN_SCORE` is not universal. It depends on the embedding model and the documents. `scripts/check_retrieval_scores.py` prints raw scores for sample questions to help tune it.

---

## API reference

Interactive documentation is generated at `/docs`. This is a summary.

### Health

`GET /health` returns `{"status": "ok", ...}`.

### Document ingestion

**Upload:** `POST /api/v1/documents/upload` (multipart form)

| Field | Required | Notes |
|-------|----------|-------|
| `file` | yes | `.pdf` or `.txt`, up to `MAX_UPLOAD_MB` |
| `chunking_strategy` | no | `fixed` or `recursive` (default `recursive`) |
| `chunk_size` | no | Characters per chunk |
| `chunk_overlap` | no | Must be smaller than `chunk_size` |

```bash
curl -F "file=@sample.txt" \
     -F "chunking_strategy=recursive" \
     -F "chunk_size=200" -F "chunk_overlap=30" \
     http://127.0.0.1:8000/api/v1/documents/upload
```

Returns `201` with the stored metadata (`id`, `filename`, `file_type`, `file_size_bytes`, `chunking_strategy`, `chunk_size`, `chunk_overlap`, `chunk_count`, `status`, `error_message`, `created_at`). Ingestion is synchronous, so `status` is already `completed` (or `failed`) in the response.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/documents?limit=50&offset=0` | List documents, newest first |
| `GET /api/v1/documents/{id}` | One document's metadata |
| `DELETE /api/v1/documents/{id}` | Delete the document and all of its vectors (`204`) |

### Chat

**Send a message:** `POST /api/v1/chat`

```json
{
  "session_id": "optional-conversation-id",
  "message": "How long do I have to ask for a refund?"
}
```

Leave `session_id` out (or empty) to start a new conversation. The response contains the generated id, so send it back on the next message. Session ids may use letters, digits, `-` and `_`, up to 64 characters.

```json
{
  "session_id": "3f2b8c1d9e7a4b5c8d6e1f0a2b3c4d5e",
  "answer": "You have 30 days from the date of purchase [1].",
  "intent": "question",
  "answered": true,
  "sources": [
    {
      "document_id": "8ab12f27-7695-4837-87b3-d1b90de5a10f",
      "filename": "sample.txt",
      "chunk_index": 0,
      "score": 0.745
    }
  ],
  "booking": null,
  "booking_status": null
}
```

- `intent` is `question` or `booking`.
- `answered` is `false` when the documents did not contain the answer (the reply is a fixed refusal and `sources` is empty).
- `sources` lists the chunks behind an answer. `chunk_index` starts at 0, and the `[1]`-style numbers in the answer refer to the order of this list.
- `booking_status` is one of `needs_info`, `confirmed`, `cancelled`, `not_in_progress`.

**A booking conversation:**

```text
User: I'd like to book an interview
Bot:  I'd be happy to book an interview. Please send me your full name, your email
      address, the interview date and the interview time.
User: I'm Asha Karki, asha.karki@example.com
Bot:  Thanks, I've noted that. I still need the interview date and the interview time.
User: Tomorrow at 2:30 pm
Bot:  You're all set, Asha! I've booked your interview for Wednesday, 28 October 2026
      at 14:30 and saved it under asha.karki@example.com.
```

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/chat/{session_id}/history` | Stored conversation (`404` if the session is unknown or expired) |
| `DELETE /api/v1/chat/{session_id}` | Delete a conversation and any booking in progress (`204`) |
| `GET /api/v1/bookings?limit=50&offset=0` | List stored interview bookings, newest first |

### Error responses

Errors are JSON of the form `{"detail": "<message>"}`.

| Status | When |
|--------|------|
| `404` | Unknown document or session |
| `413` | Upload larger than `MAX_UPLOAD_MB` |
| `415` | File type is not `.pdf` or `.txt` |
| `422` | Invalid input: bad chunk settings, bad session id, empty or unreadable file, scanned PDF without text, password-protected PDF |
| `429` | The Gemini quota is exhausted. A `Retry-After` header (seconds) is included when Google provides one. |
| `502` | An upstream service (Gemini or Qdrant) failed |

---

## How it works

### Architecture

```text
┌───────────────────────────────────────────────────────┐
│ API layer       FastAPI routers, request/response     │  HTTP only
│                 schemas                               │
├───────────────────────────────────────────────────────┤
│ Service layer   IngestionService, ChatService,        │  workflows
│                 BookingService, BookingExtractor      │
├───────────────────────────────────────────────────────┤
│ RAG / memory    loaders, chunkers, embeddings,        │  domain logic
│                 vector store, retriever, rewriter,    │
│                 pipeline, Redis memory                │
├───────────────────────────────────────────────────────┤
│ Repositories    SQLAlchemy queries only               │  data access
├───────────────────────────────────────────────────────┤
│ Infrastructure  PostgreSQL · Qdrant · Redis · Gemini  │
└───────────────────────────────────────────────────────┘
```

Each layer only talks to the layer below it. Route handlers parse HTTP and call a service, with no business logic of their own. Shared clients (Gemini, Qdrant, Redis, the database engine) are created once in the app lifespan, and per-request objects are provided through FastAPI dependencies, which tests replace with fakes.

### Ingestion flow

```text
upload ─► validate type and size ─► extract text (pypdf / UTF-8)
       ─► chunk (fixed | recursive) ─► save document row (status "processing")
       ─► embed chunks in batches ─► upsert vectors + payload into Qdrant
       ─► mark row "completed" with chunk_count
```

Problems with the file (wrong type, too large, empty, bad chunk settings) are rejected before anything is saved. If embedding or Qdrant fails after the row exists, the partial vectors are deleted and the row is marked `failed` with the reason.

**Chunking strategies**

| Strategy | Behaviour | Best for |
|----------|-----------|----------|
| `fixed` | Cuts every `chunk_size` characters, stepping back by `chunk_overlap` | Predictable, simple text |
| `recursive` | Splits on paragraph, then line, then sentence, then word, then raw characters, keeping pieces under `chunk_size` and carrying overlap forward | Natural prose, keeps ideas intact |

Both implement one interface (`BaseChunker`), and a factory selects them by name, so a third strategy needs one class and one dictionary entry.

### Chat flow

```text
message ─► load history + booking draft (Redis)
        ─► LLM reads the message: intent + any booking details (structured JSON)
           │
           ├─ booking / cancel ─► BookingService validates, updates the draft,
           │                      saves to PostgreSQL when complete
           │
           └─ question ─► rewrite follow-up into a standalone question (LLM)
                        ─► embed question ─► Qdrant top-k search with score cutoff
                        ─► no relevant chunk? fixed refusal (no LLM call)
                        ─► otherwise numbered context + question ─► LLM answers
        ─► store the turn in Redis ─► return answer, sources, booking info
```

The number of Gemini calls per message depends on the path: a booking message costs one call, a first question costs two, and a follow-up question costs three (intent, rewrite, answer). A question with no relevant chunk skips the answer call.

**Redis keys**

| Key | Type | Purpose |
|-----|------|---------|
| `chat:{session_id}` | list of JSON messages | History, trimmed to `CHAT_HISTORY_MAX_MESSAGES`, TTL refreshed on every write |
| `booking_state:{session_id}` | JSON | Booking details collected so far, with TTL |

**Qdrant**

One collection (`QDRANT_COLLECTION`), cosine distance, vector size `EMBEDDING_DIMENSION`. Each point carries `document_id`, `filename`, `chunk_index`, `text` and `strategy` as payload, and `document_id` has a payload index so per-document filtering and deletion stay fast.

**PostgreSQL**

| Table | Contents |
|-------|----------|
| `documents` | Filename, type, size, chunking strategy and sizes, chunk count, status, error message, timestamps |
| `interview_bookings` | Session id, name, email, date, time, timestamp |

Primary keys are UUIDs generated by the application. Schema changes go through Alembic migrations.

---

## Project structure

```text
rag-interview-backend/
├── app/
│   ├── main.py                    # app factory, lifespan (client setup), error handler
│   ├── core/
│   │   ├── config.py              # typed settings from environment
│   │   ├── exceptions.py          # AppError hierarchy with HTTP status codes
│   │   └── logging.py
│   ├── api/
│   │   ├── deps.py                # dependency providers
│   │   └── v1/
│   │       ├── router.py
│   │       └── endpoints/         # health, documents, chat, bookings
│   ├── schemas/                   # Pydantic request/response and LLM output models
│   ├── models/                    # SQLAlchemy ORM models and enums
│   ├── db/                        # declarative base, async engine and session factory
│   ├── repositories/              # database access (documents, bookings)
│   ├── rag/
│   │   ├── loaders.py             # PDF/TXT text extraction
│   │   ├── chunking/              # base, fixed, recursive, factory
│   │   ├── embeddings.py          # Gemini embeddings: batching, retries, validation
│   │   ├── vector_store.py        # Qdrant wrapper
│   │   ├── llm.py                 # Gemini text and structured generation
│   │   ├── retry.py               # backoff and quota-error helpers
│   │   ├── prompts.py             # all prompt templates
│   │   ├── query_rewriter.py
│   │   ├── retriever.py
│   │   └── pipeline.py            # custom RAG orchestration
│   ├── memory/
│   │   └── redis_memory.py        # chat history and booking draft storage
│   └── services/
│       ├── ingestion_service.py
│       ├── chat_service.py        # routes a message to booking or RAG
│       ├── booking_extractor.py   # LLM: intent and booking details
│       ├── booking_service.py     # validation, draft handling, persistence
│       └── booking_validation.py
├── alembic/                       # migrations
├── scripts/                       # manual checks against the real services
├── tests/
│   ├── unit/
│   └── integration/
├── alembic.ini
├── pyproject.toml                 # ruff, mypy and pytest settings
├── requirements.txt
└── .env.example
```

---

## Design decisions

- **Custom RAG pipeline.** Rewrite, retrieve, build the prompt, and generate are separate classes, each testable on its own. Nothing hides retrieval and generation behind a single prebuilt chain.
- **Query rewriting for multi-turn.** A follow-up such as "and for digital products?" means nothing to a vector search on its own. The rewriter uses recent history to turn it into a standalone question first. If rewriting fails, the original question is used instead.
- **History is used only for rewriting.** The answering step sees the retrieved context and the standalone question, not earlier replies, so the model cannot answer out of its own previous messages.
- **Strict grounding.** If no chunk reaches `RETRIEVAL_MIN_SCORE`, the pipeline returns a fixed refusal without calling the LLM. The prompt also tells the model to refuse when the excerpts do not contain the answer, and to cite excerpts by number.
- **The LLM understands, code decides.** For bookings the model only turns free text into structured fields (JSON schema output). Validation (email syntax, real calendar date, not in the past, valid time) and saving are done by deterministic code. Invalid values are reported and never stored.
- **Replies for bookings are written by code.** A confirmation message can only appear after the database write succeeded.
- **Timezone-aware bookings.** "Tomorrow" and "is this time in the past" use `BOOKING_TIMEZONE`, not the server's clock.
- **A question in the middle of a booking does not lose the booking.** The question is answered from the documents, the draft stays in Redis, and a short reminder is appended.
- **Failure handling across two stores.** Vectors and metadata live in different systems with no shared transaction, so a failed ingestion deletes its partial vectors and records `failed`. Deleting a document removes the vectors first and the row second, so a failure can be retried and never leaves vectors with no record.
- **Idempotent upserts.** Qdrant point ids are derived from the document id and chunk index, so re-ingesting a document overwrites instead of duplicating.
- **Quota-aware retries.** Temporary errors are retried with exponential backoff and jitter. A daily quota is not retried, because it cannot reset within a request, and surfaces as HTTP 429 with `Retry-After`.
- **Fail fast at startup.** The app checks Qdrant and Redis when it starts and validates every setting, including the timezone name.
- **Session ids are validated** (`[A-Za-z0-9_-]{1,64}`) because they become part of Redis keys.
- **Typed, layered code.** `mypy --strict`, `ruff`, Pydantic models at every boundary, and `Protocol`-free simple interfaces where a swap is realistic (chunkers).

---

## Testing and quality checks

```bash
ruff check .
mypy
pytest -q
```

The test suite needs no network access and uses no API quota: Gemini is replaced by fakes, Qdrant runs in its in-process mode, Redis is `fakeredis`, and API tests override dependencies. It covers chunking edge cases, loaders, retry and quota handling, the vector store, ingestion (including failure cleanup), the Redis memory, the booking rules (multi-turn, corrections, timezones, retry after a database failure), the RAG pipeline, and the HTTP layer.

`scripts/` contains manual checks that call the **real** services and therefore use API quota. Examples:

| Script | What it checks |
|--------|----------------|
| `check_gemini.py`, `check_qdrant.py`, `check_postgres.py`, `check_redis.py` | Each connection |
| `check_embedding_service.py`, `check_vector_store.py` | Embeddings and Qdrant end to end |
| `check_rag_pipeline.py` | Real question answering over the ingested documents |
| `check_retrieval_scores.py` | Raw scores, for tuning `RETRIEVAL_MIN_SCORE` |
| `check_booking_flow.py` | Real multi-turn booking, cleans up after itself |

Run any of them from the project root with `python -m scripts.<name>`.

---

## Known limitations

- **Free-tier Gemini quotas are small.** A single model can allow only a few dozen requests per day, and one chat message can use up to three calls. When the quota is used up the API returns `429` with a message and a `Retry-After` header.
- **Ingestion is synchronous.** A large PDF keeps the upload request open until embedding finishes. A background worker (Celery, RQ, or FastAPI background tasks with a status endpoint) is the natural next step.
- **Small talk is treated as a question.** "Thanks!" or "hello" has no booking intent, retrieves nothing relevant, and gets the refusal message. A dedicated small-talk intent would fix this.
- **Scanned PDFs are not supported.** `pypdf` reads the text layer only, so image-only PDFs are rejected with a clear error. OCR would be needed.
- **Duplicate uploads are not detected.** The same file uploaded twice appears twice and can crowd the top results. A content hash would prevent it.
- **Chunk sizes are measured in characters, not tokens.**
- **The retrieval cutoff was calibrated on a small sample.** Re-tune `RETRIEVAL_MIN_SCORE` for your own documents.
- **Prompt injection is only partly mitigated.** The prompts tell the model to treat document text and user text as data, but that is a mitigation, not a guarantee.
- **Booking emails are checked for syntax only.** There is no confirmation email and no slot-conflict check.
- **No authentication or rate limiting**, since they were out of scope. There is also no per-session locking, so two simultaneous messages in one session could interleave their history writes.
- **Conversations are kept in Redis for `CHAT_TTL_SECONDS`**, including any personal details typed into the chat.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---------|----------------------|
| `429` with "daily free-tier limit" | The model's daily quota is used up. Wait for the time in the message, enable billing, or set another model in `GEMINI_LLM_MODEL`, then restart. |
| App refuses to start, naming a setting | That variable is missing or invalid in `.env`. The error lists only field names, never values. |
| `Collection ... has vector size X, expected Y` | `EMBEDDING_DIMENSION` does not match the existing collection. Use a new `QDRANT_COLLECTION` name, or delete the old collection. Vectors from different embedding models must never be mixed. |
| `Could not parse SQLAlchemy URL` | `DATABASE_URL` must start with `postgresql+asyncpg://`, contain no quotes or spaces, and have special characters in the password percent-encoded. |
| `ZoneInfoNotFoundError` on Windows | Run `pip install tzdata`, since Windows has no built-in timezone database. |
| Hosted PostgreSQL needs SSL | Append `?ssl=require` to `DATABASE_URL` (the asyncpg driver does not understand `sslmode`). |
| Changed `.env` but nothing changed | `uvicorn --reload` watches Python files only. Stop and restart the server. |
| Swagger returns 422 on upload | Swagger can send empty strings for untouched optional form fields. Fill `chunk_size` and `chunk_overlap` with numbers, or leave them out. |
