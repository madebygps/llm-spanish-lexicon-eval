"""Evaluation utilities for calculating accuracy metrics."""

from storage import load_response


def calculate_accuracy(model: str, vocabulary: list[dict], prompt_type: str = "a") -> float:
    """Calculate accuracy percentage for a model for a specific prompt type"""
    correct_count = 0
    total_count = len(vocabulary)
    
    for entry in vocabulary:
        word = entry["word"]
        response_data = load_response(model, word)
        
        # Check for judgment based on prompt type
        if prompt_type == "a":
            judgment_field = response_data.get("judgment_a")
        else:  # prompt_type == "b"
            judgment_field = response_data.get("judgment_b")
            
        if judgment_field == "correct":
            correct_count += 1
    
    return (correct_count / total_count) * 100 if total_count > 0 else 0