# Contributing to RMIT RAG

Welcome! We're glad you want to contribute to the RMIT RAG project. This guide will help you get started.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nabinballa/RMIT_RAG.git
   cd RMIT_RAG
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

## Development Workflow

### Branch Strategy
- **`dev`** - Default development branch (where most work happens)
- **`main`** - Stable releases

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and test them:**
   ```bash
   # Test the CLI
   make i
   make a QUESTION="Test question"
   
   # Test the web interface
   make web
   # Open http://localhost:3000
   ```

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

4. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request:**
   - Go to [GitHub](https://github.com/nabinballa/RMIT_RAG)
   - Click "New Pull Request"
   - Select your branch → `dev`
   - Add a description of your changes

## Project Structure

```
rmit-rag/
├── api/                 # Web interface (Flask)
├── scripts/             # CLI entry points
├── src/rmit_rag/        # Core RAG logic
├── data/                # CSV files for ingestion
├── Makefile            # Build commands
└── README.md           # Main documentation
```

## Key Commands

```bash
make i                  # Build vector index
make a                  # Interactive CLI
make a QUESTION="..."   # Single question
make web                # Start web server
```

## Adding New Features

### Web Interface
- Add new routes in `api/app.py`
- Update templates in `api/templates/`

### RAG Pipeline
- Core logic in `src/rmit_rag/`
- Follow the existing interfaces in `interfaces.py`

### CLI Tools
- Add new scripts in `scripts/`
- Update `Makefile` with new targets

## Data Format

CSV files in `data/` should have columns:
- `question` - The question text
- `answer` - The answer text

## Testing

Before submitting changes:
1. Test with existing data: `make i && make a`
2. Test web interface: `make web`
3. Check for linting errors
4. Ensure README is updated if needed

## Questions?

- Check the [README.md](README.md) for detailed usage
- Open an issue for bugs or feature requests
- Ask questions in team chat/discussions

Happy coding! 🚀
