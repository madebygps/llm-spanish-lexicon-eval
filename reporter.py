"""Reporting utilities for generating summaries and displaying results."""

import json
from rich.console import Console
from rich.table import Table

from evaluator import calculate_accuracy
from storage import load_response


def generate_summary(models: list[str], vocabulary: list[dict]):
    """Generate summary.json and display results table"""
    summary = {}
    
    for model in models:
        accuracy_a = calculate_accuracy(model, vocabulary, "a")
        accuracy_b = calculate_accuracy(model, vocabulary, "b")
        summary[model] = {
            "prompt_a_accuracy": accuracy_a,
            "prompt_b_accuracy": accuracy_b
        }
    
    # Save summary.json
    with open('summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Display results table
    console = Console()
    table = Table(title="Model Performance Summary")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Prompt A Accuracy (%)", style="magenta", justify="right")
    table.add_column("Prompt A Correct", style="green", justify="right")
    table.add_column("Prompt B Accuracy (%)", style="blue", justify="right")
    table.add_column("Prompt B Correct", style="yellow", justify="right")

    # Compute number of correct definitions per model for both prompts
    correct_counts_a: dict[str, int] = {}
    correct_counts_b: dict[str, int] = {}
    
    for model in summary.keys():
        correct_a = 0
        correct_b = 0
        for entry in vocabulary:
            word = entry["word"]
            response_data = load_response(model, word)
            
            # Check prompt A judgment
            if response_data.get("judgment_a") == "correct":
                correct_a += 1
            
            # Check prompt B judgment
            if response_data.get("judgment_b") == "correct":
                correct_b += 1
                
        correct_counts_a[model] = correct_a
        correct_counts_b[model] = correct_b
    
    for model, accuracies in summary.items():
        table.add_row(
            model, 
            f"{accuracies['prompt_a_accuracy']:.1f}%",
            str(correct_counts_a.get(model, "-")),
            f"{accuracies['prompt_b_accuracy']:.1f}%",
            str(correct_counts_b.get(model, "-"))
        )
    
    console.print(table)