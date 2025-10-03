## RMIT RAG – v1 Baseline (Local, Ollama + Chroma)

Local-first RAG baseline using:
- Ollama for LLM inference (default `mistral` – Mistral 7B Instruct)
- SentenceTransformers for embeddings (default `all-MiniLM-L6-v2`)
- Chroma as the vector store (on-disk)

This repo provides a simple CLI workflow: ingest Q&A CSVs into Chroma, then query via a Make-driven interface. It mirrors the clarity of the Solara-DS docs, adapted to a local, non-Docker setup.

---

## 1. Project Structure

```bash
rmit-rag/
├── api/
│   ├── app.py             # Flask web server
│   └── templates/
│       └── index.html     # Web frontend interface
│
├── scripts/
│   ├── build_index.py     # Ingest CSVs → documents + metadata → Chroma
│   └── ask.py             # Query pipeline (retrieval + generation)
│
├── src/rmit_rag/
│   ├── __init__.py
│   ├── config.py          # Settings: chroma dir, ollama model
│   ├── data_loader.py     # CSV loader + Q&A → documents/metadatas
│   ├── interfaces.py      # Protocols for plug-and-play components
│   ├── ingestion.py       # Ingestion helper (embed + upsert)
│   ├── embedder.py        # SentenceTransformers wrapper
│   ├── preprocess.py      # Optional cleaning utilities
│   ├── vector_store.py    # Chroma wrapper
│   └── rag.py             # RAGPipeline orchestration
│
├── data/                  # Put your CSVs here (question,answer)
├── Makefile               # Make targets: index (i), ask (a), web
├── requirements.txt
└── README.md
```

👉 Summary:
- `api/` = web server and frontend
- `scripts/` = CLI entrypoints
- `src/` = core reusable logic
- `data/` = your Q&A CSVs

---

## 2. Install Ollama

Install and run Ollama, then pull a model (default used is `mistral`).

```bash
# macOS (Homebrew)
brew install ollama
brew services start ollama

# Pull the default model (Mistral 7B Instruct)
ollama pull mistral
```

If you prefer another model (e.g., `llama3`), set `OLLAMA_MODEL` accordingly (see Environment below).

---

## 3. Setup

```bash
cd /Users/nabin/Desktop/rmit-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure your CSVs live under `./data` and have exactly these columns:
- `question`
- `answer`

Example: `data/myki.csv`

---

## 4. Quick Start Flow

```bash
# 1) Build the vector index (auto-discovers *.csv in ./data)
make i

# 2) Ask a question (top-K retrieval)
make a QUESTION="How do I get a travel pass?" K=5
```

Defaults:
- `COLLECTION=combined_docs`
- `QA_MODE=concat` (store combined question+answer text)
- `K=5`

---

## 5. Make Targets

```bash
make i                   # Ingest CSVs from ./data (or DATA_DIR)
make a QUESTION="..."    # Ask a question with retrieval (K default 5)
make a                   # Interactive prompt (REPL): Enter question (or 'exit' to quit)
make web                 # Start web server (default port 5000)
```

Advanced options:
```bash
# Provide explicit CSVs with labels (label = source tag in metadata)
make i QA="./data/myki.csv:myki,./data/housing.csv:housing" CLEAR=1

# Enable preprocessing (cleaning)
make i PREPROCESS=1 PRE_TO_LOWER=1 PRE_STRIP_CONTROLS=1 PRE_NORMALIZE_SPACES=1 PRE_MIN_LENGTH=10

# Change discovery directory
make i DATA_DIR=./data

# Web server options
make web PORT=8080 FLASK_DEBUG=true
```

---

## 6. Environment

Optional environment variables (via shell or `.env`):

```ini
# Vector DB location (folder will be created if missing)
CHROMA_DIR=chroma

# Ollama model for generation
OLLAMA_MODEL=mistral

# Only relevant if reading spreadsheets elsewhere
SHEET_NAME=Sheet1

# Web server settings
PORT=5000
FLASK_DEBUG=false

# Personality settings
PERSONALITY_LEVEL=friendly  # friendly, professional, casual, enthusiastic
TEMPERATURE=0.4            # 0.1-1.0, higher = more creative

# Performance settings (for faster responses)
MAX_RESPONSE_LENGTH=512    # Limit response length (lower = faster)
CONTEXT_WINDOW=2048        # Limit context window (lower = faster)
BATCH_SIZE=32              # Embedding batch size (higher = faster for bulk operations)
```

You can set a different default model globally:

```bash
ollama pull llama3
echo 'export OLLAMA_MODEL=llama3' >> ~/.zshrc && source ~/.zshrc
```

---

## 7. Data Prep

Prepare one or more CSVs with columns `question,answer`.

Examples:
```bash
data/travel_pass.csv
data/housing.csv
data/driving_license.csv
```

Modes when converting rows to documents:
- `concat` (default): concatenates question + answer into one text field
- `answer`: uses only the answer text for embedding/retrieval

Control via `QA_MODE=concat|answer` when running `make i`.

---

## 8. Developer Notes

- Embedder: `rmit_rag.embedder.Embedder` implements `rmit_rag.interfaces.EmbedderProtocol`
- Vector DB: `rmit_rag.vector_store.VectorStore` implements `rmit_rag.interfaces.VectorStoreProtocol`
- Ingestion helper: `rmit_rag.ingestion.ingest_documents(embedder, store, documents, metadatas)`
- Converters:
  - `rmit_rag.data_loader.load_qa_csv(path)`
  - `rmit_rag.data_loader.qa_dataframe_to_documents(df, mode="concat"|"answer")`

Wire custom components by passing them into `RAGPipeline`:

```python
from rmit_rag.rag import RAGPipeline
from rmit_rag.embedder import Embedder
from rmit_rag.vector_store import VectorStore
from rmit_rag.config import settings

embedder = Embedder("all-MiniLM-L6-v2")
store = VectorStore("combined_docs", persist_directory=settings.chroma_dir)
pipeline = RAGPipeline("combined_docs", embedder=embedder, store=store)
```

---

## 9. Performance Optimization

### Speed Up Responses:
```bash
# Faster responses (shorter, more focused)
export MAX_RESPONSE_LENGTH=256
export CONTEXT_WINDOW=1024
make web

# Use smaller embedding model for faster encoding
# Edit src/rmit_rag/embedder.py and change model to "all-MiniLM-L12-v2"
```

### Hardware Acceleration:
- **Apple Silicon**: Automatically uses MPS (Metal Performance Shaders)
- **NVIDIA GPU**: Automatically uses CUDA if available
- **CPU**: Optimized with threading and caching

### Caching:
- Embedding models are cached globally (no reloading between requests)
- Vector store uses optimized queries
- Ollama model stays loaded in memory

## 10. Troubleshooting

- Ollama not running
  - Start: `brew services start ollama` (macOS) or run `ollama serve` in a separate terminal
  - Pull a model: `ollama pull mistral`

- No CSVs found
  - Ensure files exist under `./data` and have columns `question,answer`

- Rebuild from scratch
  - Use `CLEAR=1` with `make i` to reset the collection before ingest

- Change collection name
  - Pass `COLLECTION=your_collection` to both `make i` and `make a`

- Slow responses
  - Reduce `MAX_RESPONSE_LENGTH` and `CONTEXT_WINDOW`
  - Use smaller embedding model
  - Ensure Ollama is running locally (not over network)

---

## 11. Web Interface

Start the web server:
```bash
make web
```

Then open your browser to `http://localhost:5000` for a clean web interface to ask questions.

Features:
- Clean, modern UI with chat interface
- Real-time status checking
- Adjustable retrieval count (K)
- **4 Personality modes**: Friendly 😊, Professional 👔, Casual 😎, Enthusiastic 🎉
- **Creativity slider**: Adjust response creativity (0.1-1.0)
- Error handling and loading states
- Message history and typing indicators

---

## 12. Examples

```bash
# Minimal ingest and query
make i
make a QUESTION="How do I renew a driving license?" K=5

# Label sources explicitly and clean text
make i QA="./data/myki.csv:myki,./data/housing.csv:housing" CLEAR=1 \
  PREPROCESS=1 PRE_TO_LOWER=1 PRE_STRIP_CONTROLS=1 PRE_NORMALIZE_SPACES=1 PRE_MIN_LENGTH=10

# Start web interface
make web PORT=8080

# Set personality and temperature
export PERSONALITY_LEVEL=enthusiastic
export TEMPERATURE=0.6
make web
```

