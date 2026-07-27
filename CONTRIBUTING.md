# Contributing to Self-Healing RAG

Thank you for your interest in contributing to **Self-Healing RAG**! We welcome bug reports, feature enhancements, documentation improvements, and architectural suggestions.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/self-healing-rag.git
   cd self-healing-rag
   ```

2. **Create a virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies in editable mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Environment setup**:
   ```bash
   cp .env.example .env
   ```

5. **Run tests**:
   ```bash
   pytest
   ```

## Pull Request Guidelines

- Ensure all new features include unit tests in `tests/`.
- Keep code clean, explicit, and annotated with type hints and docstrings.
- Verify existing tests pass before submitting your PR.
