"""Storage utilities for managing response files."""

import json
from pathlib import Path


def save_response(model: str, word: str, correct_definition: str, model_response_a: str = "", model_response_b: str = "", judgment_a: str = "", judgment_b: str = ""):
    """Save model response to output directory"""
    output_dir = Path(f"output/{model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing data if it exists to preserve existing responses
    file_path = output_dir / f"{word}.json"
    response_data = {}
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
    
    # Update with new data
    response_data.update({
        "word": word,
        "correct_definition": correct_definition
    })
    
    if model_response_a:
        response_data["model_response_a"] = model_response_a
    if model_response_b:
        response_data["model_response_b"] = model_response_b
    if judgment_a:
        response_data["judgment_a"] = judgment_a
    if judgment_b:
        response_data["judgment_b"] = judgment_b
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)


def load_response(model: str, word: str) -> dict:
    """Load existing response from output directory"""
    file_path = Path(f"output/{model}/{word}.json")
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def update_response_judgment(model: str, word: str, judgment_a: str = "", judgment_b: str = ""):
    """Update existing response with judgment"""
    response_data = load_response(model, word)
    
    if judgment_a:
        response_data["judgment_a"] = judgment_a
    if judgment_b:
        response_data["judgment_b"] = judgment_b
    
    output_dir = Path(f"output/{model}")
    file_path = output_dir / f"{word}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)