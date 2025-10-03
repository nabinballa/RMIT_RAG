#!/bin/bash

# RMIT RAG Setup Script for Team Members
# Run this script to set up the development environment

echo "🚀 Setting up RMIT RAG development environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed. Please install Ollama first:"
    echo "   macOS: brew install ollama"
    echo "   Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "   Then run: ollama pull mistral"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# RMIT RAG Configuration
CHROMA_DIR=chroma
OLLAMA_MODEL=mistral
SHEET_NAME=Sheet1
PORT=3000
FLASK_DEBUG=false
EOF
    echo "✅ Created .env file with default settings"
fi

# Check if Ollama model is available
echo "🤖 Checking Ollama model..."
if ollama list | grep -q "mistral"; then
    echo "✅ Mistral model is available"
else
    echo "📥 Pulling Mistral model (this may take a while)..."
    ollama pull mistral
fi

echo ""
echo "🎉 Setup complete! Here's how to get started:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Build the index:"
echo "   make i"
echo ""
echo "3. Start the web interface:"
echo "   make web"
echo ""
echo "4. Or use the CLI:"
echo "   make a"
echo ""
echo "📖 See README.md for more details!"
