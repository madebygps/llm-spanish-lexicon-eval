"""Data loading utilities for Spanish lexicon evaluation."""

import json


def load_models() -> list[str]:
    """Load active models from models_list.txt (excluding # commented lines)"""
    models = []
    with open('suite/models_list.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                models.append(line)
    return models


def load_prompts() -> dict:
    """Load prompts from prompts.json"""
    with open('suite/prompts.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_vocabulary() -> list[dict]:
    """Load vocabulary from vocabulary_short.json"""
    with open('suite/vocabulary_short.json', 'r', encoding='utf-8') as f:
        return json.load(f)