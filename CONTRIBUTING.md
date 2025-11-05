# Contributing

## Development Setup

1. Clone the repository
2. Create virtual environment: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Download spaCy model: `python -m spacy download en_core_web_sm`

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all functions and classes
- Use logging instead of print statements

## Testing

Run tests:
```bash
python tests/test_basic_functionality.py
python tests/check_setup.py
```

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Run tests
4. Update documentation if needed
5. Submit pull request

