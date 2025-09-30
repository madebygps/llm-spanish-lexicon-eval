# LLM Spanish Lexicon Evaluation

A comprehensive evaluation framework for testing LLM understanding of Spanish vocabulary across different prompting strategies.

## 🎯 Overview

This project evaluates how well different Large Language Models understand Spanish vocabulary by:
1. **Prompt A**: Asking models to define Spanish words
2. **Prompt B**: Asking models to use words in context (two sentences)
3. Using GPT-5 as a judge to evaluate response correctness
4. Generating comparative accuracy reports

## 📋 Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency management
- [Ollama](https://ollama.com/) installed and running locally
- OpenAI API key with GPT-5 access

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/madebygps/llm-spanish-lexicon-eval.git
cd llm-spanish-lexicon-eval

# Install dependencies
uv sync

# Set up your OpenAI API key as environment variable
export OPENAI_API_KEY='your-api-key-here'

# Run the evaluation
uv run python main.py
```

This will:
1. Load active models from `suite/models_list.txt`
2. Test each model with both prompt types on vocabulary words
3. Use GPT-5 to judge all responses
4. Generate `summary.json` and display a Rich table with results

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
└── tests/                  # Comprehensive test suite (93% coverage)
```

## 🧪 Testing

This project has a comprehensive test suite with **93% code coverage**.

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=. --cov-report=html
# Then open htmlcov/index.html

# Run specific test file
uv run pytest tests/test_data_loader.py -v
```

| Module | Coverage |
|--------|----------|
| `data_loader.py` | 100% |
| `storage.py` | 100% |
| `evaluator.py` | 100% |
| `model_client.py` | 100% |
| `reporter.py` | 100% |
| **Overall** | **93%** |

## 📝 Configuration

### Models List (`suite/models_list.txt`)

Select which models to evaluate by uncommenting them (remove `#`):

```
#solar:10.7b
#mixtral:latest
gemma3:12b       
#mistral:latest 
llama3.1:latest
```

**To run different models:**
1. Comment out models you don't want to test with `#`
2. Uncomment models you want to test (remove `#`)
3. Make sure the model is installed in Ollama: `ollama pull model-name`

### Prompts (`suite/prompts.json`)

Two prompt strategies are used:

```json
{
  "prompt_a": "Dime la definición de la palabra '{word}'.",
  "prompt_b": "Escribe dos frases, una con la palabra '{word}', y otra que no contenga esa palabra, pero que esté relacionada con la primera y complemente su significado."
}
```

### Vocabulary (`suite/vocabulary_short.json`)

Each entry contains a Spanish word and its definition:

```json
[
  {
    "word": "ardilla",
    "answer": "Mamífero roedor, de unos 20 cm de largo, de color negro rojizo por el lomo, blanco por el vientre y con cola muy poblada."
  },
  {
    "word": "corbata",
    "answer": "Prenda de adorno, especialmente masculina, consistente en una banda larga y estrecha que se coloca alrededor del cuello."
  }
]
```

**To add new vocabulary:**
1. Edit `suite/vocabulary_short.json` or `suite/vocabulary_complete.json`
2. Add entries with `word` and `answer` fields
3. The code will automatically use `vocabulary_short.json` by default

## 📊 Output

### Summary Report

After running the evaluation, a `summary.json` file is generated with accuracy metrics:

```json
{
  "gemma3:12b": {
    "prompt_a_accuracy": 60.0,
    "prompt_b_accuracy": 90.0
  },
  "llama3.1:latest": {
    "prompt_a_accuracy": 20.0,
    "prompt_b_accuracy": 40.0
  }
}
```

A Rich table is also displayed in the terminal:

```
                Model Performance Summary
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Model          ┃ Prompt A Accuracy   ┃ Prompt A        ┃ Prompt B Accuracy   ┃ Prompt B        ┃
┃                ┃ (%)                 ┃ Correct         ┃ (%)                 ┃ Correct         ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ gemma3:12b     │                60.0%│                6│                90.0%│                9│
│ llama3.1:latest│                20.0%│                2│                40.0%│                4│
└────────────────┴─────────────────────┴─────────────────┴─────────────────────┴─────────────────┘
```

### Response Files

Individual model responses are saved to `output/{model}/{word}.json`:

```json
{
  "word": "ardilla",
  "correct_definition": "Mamífero roedor, de unos 20 cm de largo...",
  "model_response_a": "Un animal pequeño que vive en los árboles...",
  "judgment_a": "correct",
  "model_response_b": "La ardilla guarda nueces para el invierno. Este comportamiento es común en roedores.",
  "judgment_b": "correct"
}
```

### Next Steps After Running

1. **Review Results**: Check `summary.json` and the terminal table to compare model performance
2. **Analyze Responses**: Examine individual response files in `output/{model}/` to understand where models succeed or fail
3. **Adjust Configuration**: 
   - Try different models by editing `suite/models_list.txt`
   - Test with more vocabulary by switching to `vocabulary_complete.json` in `data_loader.py`
   - Modify prompts in `suite/prompts.json` to test different strategies
4. **Re-run Evaluation**: The system will skip already-evaluated responses, so you can resume or add new models anytime

## 🛠️ Development

### Code Style
- Use type hints for all functions
- Follow PEP 8 guidelines
- Keep functions focused and testable
- Document with docstrings

### Adding New Features
1. Create the module file
2. Add corresponding test file in `tests/`
3. Run tests: `uv run pytest`
4. Check coverage: `uv run pytest --cov`

## 🧩 Architecture

The project follows a modular design with clear separation of concerns:

- **Data Layer** (`data_loader.py`, `storage.py`): File I/O operations
- **Model Layer** (`model_client.py`): API interactions with LLMs
- **Business Logic** (`evaluator.py`): Accuracy calculations
- **Presentation** (`reporter.py`): Output formatting
- **Orchestration** (`main.py`): Workflow coordination

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

## 📚 Resources

- [Test Suite Documentation](./tests/README.md)

## ✨ Features

- ✅ Modular, testable architecture
- ✅ 93% test coverage
- ✅ UTF-8 support for Spanish characters
- ✅ Resumable evaluation (skips existing responses)
- ✅ Multiple prompting strategies
- ✅ GPT-5 based evaluation
- ✅ Rich terminal output
- ✅ JSON-based configuration

## 📄 License

MIT License - See [LICENSE](LICENSE) for details
