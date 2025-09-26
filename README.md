## RMIT RAG

Modular RAG pipeline combining multiple CSV/XLSX sources into a Chroma vector store, with a CLI to index data and query using Ollama (e.g., llama3) and SentenceTransformers embeddings.

### Setup

1. Create and activate a virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and the model pulled
```bash
# Install from https://ollama.com
ollama pull llama3
```

### Project Structure
```
src/rmit_rag/
  __init__.py
  config.py
  data_loader.py
  embedder.py
  vector_store.py
  rag.py
scripts/
  build_index.py
  ask.py
```

### Data
Place your files under `data/`:
- `rmit_housing_full.xlsx` (Sheet1)
- `emergency.csv`
- `oshc.csv`

### Usage

Build the index:
```bash
python scripts/build_index.py --data-dir ./data --collection combined_docs
```

Ask a question:
```bash
python scripts/ask.py --collection combined_docs "What are the average shared room prices in Carlton?"
```

### Environment
Optional `.env` variables:
- `CHROMA_DIR` (default: `chroma`)
- `OLLAMA_MODEL` (default: `llama3`)
- `SHEET_NAME` (default: `Sheet1`)
