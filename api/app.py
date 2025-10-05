#!/usr/bin/env python
from __future__ import annotations
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, request, jsonify, render_template, Response, stream_template
import json
import time
from rmit_rag.rag import RAGPipeline
from rmit_rag.embedder import Embedder
from rmit_rag.vector_store import VectorStore
from rmit_rag.config import settings
from rmit_rag.personality import get_available_personalities
from rmit_rag.cache import clear_cache, get_cache_stats

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
        k = int(data.get("k", 3))  # Default to 3 for faster responses
        stream = data.get("stream", False)
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        init_pipeline()
        
        if stream:
            return Response(stream_with_question(question, k), mimetype='application/json')
        else:
            start_time = time.time()
            answer = pipeline.query(question, n_results=k)
            end_time = time.time()
            
            return jsonify({
                "question": question,
                "answer": answer,
                "k": k,
                "response_time": round(end_time - start_time, 2)
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def stream_with_question(question, k):
    """Stream the response for better perceived performance."""
    try:
        start_time = time.time()
        
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your question...'})}\n\n"
        
        # Get embeddings (fastest part)
        query_embedding = pipeline.embedder.encode([question])
        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
        
        # Query vector store
        results = pipeline.store.query(query_embeddings=query_embedding, n_results=k)
        context = " ".join(results["documents"][0]) if results and results.get("documents") else ""
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
        
        # Generate response
        answer = pipeline.query(question, n_results=k)
        end_time = time.time()
        
        # Send final result
        yield f"data: {json.dumps({'type': 'complete', 'answer': answer, 'response_time': round(end_time - start_time, 2)})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

@app.route("/api/status")
def status():
    """Health check endpoint."""
    return jsonify({
        "status": "ok", 
        "model": settings.ollama_model,
        "personality": settings.personality_level,
        "temperature": settings.temperature
    })

@app.route("/api/personalities")
def personalities():
    """Get available personality types."""
    return jsonify(get_available_personalities())

@app.route("/api/config", methods=["POST"])
def update_config():
    """Update personality and temperature settings."""
    try:
        data = request.get_json()
        
        # Update environment variables for this session
        if "personality" in data:
            os.environ["PERSONALITY_LEVEL"] = data["personality"]
        if "temperature" in data:
            os.environ["TEMPERATURE"] = str(data["temperature"])
            
        # Reinitialize pipeline with new settings
        global pipeline
        pipeline = None
        init_pipeline()
        
        return jsonify({
            "status": "ok",
            "personality": os.environ.get("PERSONALITY_LEVEL", "friendly"),
            "temperature": float(os.environ.get("TEMPERATURE", "0.4"))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cache/clear", methods=["POST"])
def clear_response_cache():
    """Clear the response cache."""
    try:
        clear_cache()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cache/stats", methods=["GET"])
def get_cache_statistics():
    """Get cache statistics."""
    try:
        stats = get_cache_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_pipeline()
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
