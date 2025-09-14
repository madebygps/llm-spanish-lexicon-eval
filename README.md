# LLM Spanish Lexicon Evaluation

A  evaluation suite for testing Spanish language comprehension and lexical knowledge in Large Language Models (LLMs). Based on Conde et al.

## 🎯 Overview

This project provides a standardized benchmark for evaluating Spanish language capabilities across multiple domains:

- **Vocabulary Definitions**: MCQ tests with plausible distractors
- **Reading Comprehension**: Context-based question answering  
- **MCQ Reasoning**: Logic and inference testing
- **Cloze Tests**: Fill-in-the-blank sentence completion
- **Everyday Q&A**: Common knowledge questions

The suite generates weighted scores across all test categories, providing both granular insights and an overall grade (A+ to F) for Spanish comprehension ability.

## 📊 Test Categories

| Test Type | Weight | Description | Example |
|-----------|--------|-------------|---------|
| **MCQ Reasoning** | 25% | Context-based logical reasoning | Given a passage, answer inference questions |
| **Reading Comprehension** | 25% | Text understanding and analysis | Extract specific information from Spanish texts |
| **Everyday Q&A** | 20% | Common knowledge questions | "¿Cuál es la capital de España?" |
| **Cloze MCQ** | 15% | Sentence completion tasks | "El perro _____ en el jardín" |
| **Vocabulary Definitions** | 15% | Word definition matching | Choose correct definition from 4 options |

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- OpenAI-compatible API endpoint (OpenAI, Ollama, etc.)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/madebygps/llm-spanish-lexicon-eval.git
cd llm-spanish-lexicon-eval
```

1. Install dependencies:

```bash
pip install -e .
```

### Running Tests

#### For Ollama (Local Models)

```bash
# Start Ollama with your preferred model
ollama pull mistral
ollama serve

# Run evaluation
python main.py
```

#### For OpenAI API

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Modify main.py to use OpenAI endpoint
python main.py
```

## 📁 Project Structure

```text
llm-spanish-lexicon-eval/
├── main.py                    # Main evaluation script
├── generate_vocab_mcq.py      # Vocabulary test generator
├── words.txt                  # Spanish word list
├── suite/                     # Test datasets
│   ├── cloze_mcq.jsonl       # Fill-in-the-blank tests
│   ├── everyday_qa.jsonl     # Common knowledge Q&A
│   ├── mcq_reasoning.jsonl   # Logic and reasoning tests
│   ├── reading_comprehension.jsonl # Text comprehension
│   └── vocabulary_complete.json # Vocabulary definitions
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## 🧪 Test Details

### Vocabulary Definitions

- **Format**: Multiple choice questions with 4 options
- **Source**: Spanish dictionary definitions with generated distractors
- **Sample Size**: 20 random questions per evaluation
- **Scoring**: Exact match or numbered response recognition

### Reading Comprehension

- **Format**: Context passage + open-ended questions
- **Evaluation**: Flexible matching (exact, substring, keyword-based)
- **Focuses**: Information extraction and text understanding

### MCQ Reasoning

- **Format**: Context + question + 4 multiple choice options
- **Tests**: Logical inference and critical thinking
- **Evaluation**: Choice matching with flexible response parsing

### Cloze MCQ

- **Format**: Sentences with missing words + 4 options
- **Tests**: Grammar, vocabulary, and context understanding
- **Examples**: "El perro _____ en el jardín" → "corre"

### Everyday Q&A

- **Format**: Direct questions requiring factual knowledge
- **Topics**: Geography, culture, common sense
- **Evaluation**: Flexible answer matching with normalization

## 📈 Scoring System

The evaluation produces both detailed per-category results and an overall weighted grade:

### Grade Scale

- **A+ (90-100%)**: Excellent Spanish comprehension
- **A (80-89%)**: Very good Spanish comprehension  
- **B (70-79%)**: Good Spanish comprehension
- **C (60-69%)**: Fair Spanish comprehension
- **D (50-59%)**: Basic Spanish comprehension
- **F (0-49%)**: Poor Spanish comprehension

### Sample Output

```text
📊 FINAL RESULTS
==================================================
MCQ Reasoning:         78.5% (weight: 25%)
Reading Comprehension: 82.1% (weight: 25%)
Everyday Q&A:          75.3% (weight: 20%)
Cloze MCQ:             91.2% (weight: 15%)
Vocabulary Definitions: 68.9% (weight: 15%)
--------------------------------------------------
🎯 OVERALL GRADE:       78.1%
📈 Grade: B - Good Spanish comprehension
```

## 🛠 Configuration

### Model Configuration

Edit `main.py` to modify:

- API endpoint URL
- Model name
- Temperature settings
- Token limits

### Test Customization

- Modify test files in `suite/` directory
- Adjust weights in the `main()` function
- Add new test categories by implementing new functions
