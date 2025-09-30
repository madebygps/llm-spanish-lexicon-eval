# LLM Spanish Lexicon Evaluation

A comprehensive evaluation framework for testing LLM understanding of Spanish vocabulary across different prompting strategies.

## 🎯 Overview

This project evaluates how well different Large Language Models understand Spanish vocabulary by:
1. **Prompt A**: Asking models to define Spanish words
2. **Prompt B**: Asking models to use words in context (two sentences)
3. Using GPT-5 as a judge to evaluate response correctness
4. Generating comparative accuracy reports

## 📊 Project Structure

```
llm-spanish-lexicon-eval/
├── data_loader.py          # Load models, prompts, and vocabulary
├── model_client.py         # Interface with Ollama and OpenAI APIs
├── storage.py              # Save/load response data
├── evaluator.py            # Calculate accuracy metrics
├── reporter.py             # Generate summaries and tables
├── main.py                 # Main orchestration script
├── suite/
│   ├── models_list.txt     # Models to evaluate (# for comments)
│   ├── prompts.json        # Prompt templates
│   └── vocabulary_*.json   # Spanish words to test
├── output/                 # Individual response files per model
└── tests/                  # Comprehensive test suite (64 tests, 93% coverage)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Ollama running locally (for LLM evaluation)
- OpenAI API key (for GPT-5 judging)

### Installation

```bash
# Clone the repository
git clone https://github.com/madebygps/llm-spanish-lexicon-eval.git
cd llm-spanish-lexicon-eval

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Running the Evaluation

```bash
# Run the full evaluation pipeline
uv run python main.py
```

This will:
1. Load active models from `suite/models_list.txt`
2. Prompt each model with both prompt types
3. Use GPT-5 to judge all responses
4. Generate `summary.json` and display results

## 🧪 Testing

This project has a comprehensive test suite with **93% code coverage**.

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=. --cov-report=html
# Then open htmlcov/index.html

# Run specific test file
uv run pytest tests/test_data_loader.py
```

### Test Coverage

| Module | Coverage |
|--------|----------|
| `data_loader.py` | 100% |
| `storage.py` | 100% |
| `evaluator.py` | 100% |
| `model_client.py` | 100% |
| `reporter.py` | 100% |
| **Overall** | **93%** |

See [TEST_SUITE_SUMMARY.md](TEST_SUITE_SUMMARY.md) for detailed test documentation.

## 📝 Configuration

### Models List (`suite/models_list.txt`)
```
# Comment out models with #
gemma3:12b
llama3.1:latest
#solar:10.7b
```

### Prompts (`suite/prompts.json`)
```json
{
  "prompt_a": "Define la palabra {word}",
  "prompt_b": "Escribe dos frases sobre {word}..."
}
```

### Vocabulary (`suite/vocabulary_short.json`)
```json
[
  {"word": "ardilla", "answer": "Un roedor pequeño..."},
  {"word": "corbata", "answer": "Una prenda de vestir..."}
]
```

## 📊 Output

### Response Files
Individual responses are saved to `output/{model}/{word}.json`:
```json
{
  "word": "ardilla",
  "correct_definition": "Un roedor pequeño...",
  "model_response_a": "...",
  "judgment_a": "correct",
  "model_response_b": "...",
  "judgment_b": "incorrect"
}
```

### Summary Report
The `summary.json` file contains accuracy metrics:
```json
{
  "gemma3:12b": {
    "prompt_a_accuracy": 80.0,
    "prompt_b_accuracy": 70.0
  },
  "llama3.1:latest": {
    "prompt_a_accuracy": 90.0,
    "prompt_b_accuracy": 85.0
  }
}
```

A Rich table is also displayed in the terminal with the results.

## 🛠️ Development

### Adding a New Module
1. Create the module file
2. Add corresponding test file in `tests/`
3. Run tests: `uv run pytest`
4. Check coverage: `uv run pytest --cov`

### Code Style
- Use type hints for all functions
- Follow PEP 8 guidelines
- Keep functions focused and testable
- Document with docstrings

## 🧩 Architecture

The project follows a modular design with clear separation of concerns:

- **Data Layer** (`data_loader.py`, `storage.py`): File I/O operations
- **Model Layer** (`model_client.py`): API interactions with LLMs
- **Business Logic** (`evaluator.py`): Accuracy calculations
- **Presentation** (`reporter.py`): Output formatting
- **Orchestration** (`main.py`): Workflow coordination

All modules are independently testable with 100% coverage.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

## 📚 Resources

- [Project Documentation](./docs/)
- [Test Suite Documentation](./tests/README.md)
- [Test Summary](./TEST_SUITE_SUMMARY.md)

## ✨ Features

- ✅ Modular, testable architecture
- ✅ 93% test coverage
- ✅ UTF-8 support for Spanish characters
- ✅ Resumable evaluation (skips existing responses)
- ✅ Multiple prompting strategies
- ✅ GPT-5 based evaluation
- ✅ Rich terminal output
- ✅ JSON-based configuration
