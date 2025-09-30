# Tests for LLM Spanish Lexicon Evaluation

This directory contains comprehensive unit tests for the Spanish lexicon evaluation project.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_data_loader.py      # Tests for data loading functions
├── test_storage.py          # Tests for storage operations
├── test_evaluator.py        # Tests for accuracy calculations
├── test_model_client.py     # Tests for AI model interactions (mocked)
└── test_reporter.py         # Tests for summary generation
```

## Running Tests

### Run all tests
```bash
uv run pytest
```

### Run with verbose output
```bash
uv run pytest -v
```

### Run specific test file
```bash
uv run pytest tests/test_data_loader.py
```

### Run specific test
```bash
uv run pytest tests/test_data_loader.py::TestLoadModels::test_load_models_filters_comments
```

### Run with coverage report
```bash
uv run pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` to view the coverage report.

### Run tests matching a pattern
```bash
uv run pytest -k "test_save"
```

## Test Coverage

The test suite covers:

### ✅ Data Loader (`test_data_loader.py`)
- Loading models from file and filtering comments
- Loading prompts JSON with placeholders
- Loading vocabulary with UTF-8 support
- Handling empty lines and whitespace

### ✅ Storage (`test_storage.py`)
- Creating and saving response JSON files
- Loading existing responses
- Updating responses with judgments
- Preserving existing data during updates
- UTF-8 character handling

### ✅ Evaluator (`test_evaluator.py`)
- Calculating accuracy for both prompts
- Handling all correct/incorrect scenarios
- Partial correctness calculations
- Edge cases (empty vocabulary, missing judgments)

### ✅ Model Client (`test_model_client.py`)
- Mocked API calls to Ollama and OpenAI
- Prompt template formatting
- Response judging logic
- Error handling for None responses

### ✅ Reporter (`test_reporter.py`)
- Summary JSON generation
- Accuracy calculations across models
- Table display (mocked)
- UTF-8 preservation

## Testing Best Practices

1. **Isolation**: Each test is independent and doesn't rely on others
2. **Mocking**: External API calls are mocked to avoid costs and network dependencies
3. **Temporary Files**: Uses `tmp_path` fixture to avoid polluting real output directories
4. **UTF-8 Testing**: Validates proper handling of Spanish characters
5. **Edge Cases**: Tests empty inputs, missing data, and error conditions

## Fixtures

Shared fixtures are defined in `conftest.py`:
- `sample_models`: Standard model names
- `sample_prompts`: Standard prompt templates
- `sample_vocabulary`: Standard vocabulary entries

## Notes

- All tests use temporary directories (`tmp_path`) to avoid interfering with actual project data
- API calls to OpenAI and Ollama are mocked to prevent actual API usage during testing
- Tests validate both happy paths and edge cases
