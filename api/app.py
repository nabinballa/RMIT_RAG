#!/usr/bin/env python
from __future__ import annotations
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, request, jsonify, render_template
from rmit_rag.rag import RAGPipeline
from rmit_rag.embedder import Embedder
from rmit_rag.vector_store import VectorStore
from rmit_rag.config import settings

app = Flask(__name__)

# Initialize RAG pipeline once at startup
pipeline = None

def init_pipeline():
    global pipeline
    if pipeline is None:
        collection = os.getenv("COLLECTION", "combined_docs")
        embedder = Embedder("all-MiniLM-L6-v2")
        store = VectorStore(collection, persist_directory=settings.chroma_dir)
        pipeline = RAGPipeline(collection, embedder=embedder, store=store)

@app.route("/")
def index():
    """Serve the chat interface."""
    return render_template("chat.html")

@app.route("/old")
def old_interface():
    """Serve the old frontend interface."""
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask_question():
    """Ask a question to the RAG system."""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        k = int(data.get("k", 5))
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        init_pipeline()
        answer = pipeline.query(question, n_results=k)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "k": k
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status")
def status():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": settings.ollama_model})

if __name__ == "__main__":
    init_pipeline()
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
