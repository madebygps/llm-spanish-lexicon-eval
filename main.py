
"""Spanish lexicon evaluation orchestration."""

from tqdm import tqdm
from rich.console import Console

from data_loader import load_models, load_prompts, load_vocabulary
from model_client import prompt_model, judge_response, judge_response_b
from storage import save_response, load_response, update_response_judgment
from reporter import generate_summary


def main():
    # Load data
    models = load_models()
    prompts = load_prompts()
    vocabulary = load_vocabulary()
    
    prompt_template_a = prompts["prompt_a"]
    prompt_template_b = prompts["prompt_b"]
    
    console = Console()
    console.print(f"[bold green]Starting evaluation with {len(models)} models and {len(vocabulary)} words[/bold green]")
    
    # Step 2: Prompt models with both prompts
    for model in models:
        console.print(f"[bold blue]Processing model: {model}[/bold blue]")
        
        for entry in tqdm(vocabulary, desc=f"Prompting {model}"):
            word = entry["word"]
            correct_definition = entry["answer"]
            
            response_data = load_response(model, word)
            
            # Process prompt A
            if not response_data.get("model_response_a"):
                model_response_a = prompt_model(word, model, prompt_template_a)
                save_response(model, word, correct_definition, model_response_a=model_response_a)
            
            # Process prompt B
            if not response_data.get("model_response_b"):
                model_response_b = prompt_model(word, model, prompt_template_b)
                save_response(model, word, correct_definition, model_response_b=model_response_b)
    
    # Step 3: Judge responses for both prompts
    console.print("[bold yellow]Starting judgment phase...[/bold yellow]")
    
    for model in models:
        console.print(f"[bold blue]Judging responses for: {model}[/bold blue]")
        
        for entry in tqdm(vocabulary, desc=f"Judging {model}"):
            word = entry["word"]
            correct_definition = entry["answer"]
            
            response_data = load_response(model, word)
            if not response_data:
                continue
            
            # Judge prompt A response
            if not response_data.get("judgment_a"):
                model_response_a = response_data.get("model_response_a")
                if model_response_a:
                    judgment_a = judge_response(word, correct_definition, model_response_a)
                    update_response_judgment(model, word, judgment_a=judgment_a)
            
            # Judge prompt B response
            if not response_data.get("judgment_b"):
                model_response_b = response_data.get("model_response_b")
                if model_response_b:
                    judgment_b = judge_response_b(word, correct_definition, model_response_b)
                    update_response_judgment(model, word, judgment_b=judgment_b)
    
    # Step 4: Generate summary
    console.print("[bold green]Generating summary...[/bold green]")
    generate_summary(models, vocabulary)


if __name__ == "__main__":
    main()