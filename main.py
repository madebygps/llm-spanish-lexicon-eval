
import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from rich.console import Console
from rich.table import Table


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


def prompt_model(word: str, model: str, prompt_template: str) -> str:
    """Prompt a model via OLAMA using OpenAI client"""
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )
    
    prompt = prompt_template.format(word=word)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""


def judge_response(word: str, correct_definition: str, model_response: str) -> str:
    """Use GPT-5 to judge if the model response is correct or incorrect"""
    client = OpenAI()  # Uses standard OpenAI API
    
    judge_prompt = f"""
    Evalúa si la definición propuesta es suficientemente correcta (no necesita ser literal) para la palabra indicada.

    Criterios para marcar correct:
    - Captura el núcleo semántico esencial aunque use sinónimos o parafrasee.
    - Coincide la categoría gramatical y el sentido principal válido en uso general.
    - Puede omitir detalles secundarios siempre que no distorsione el significado central.
    - Si la palabra tiene varios sentidos, acepta uno legítimo y común salvo que la definición de referencia delimite claramente otro sentido específico.

    Marca incorrect si:
    - Cambia el significado central o selecciona un sentido no pertinente frente a uno claramente indicado.
    - Es tan vaga o general que podría aplicarse a muchos otros términos sin identificar este.
    - Es demasiado estrecha o añade rasgos críticos que no forman parte del significado.
    - Omite un componente indispensable que altera el concepto.
    - Introduce información falsa, confusa, o mezcla con otro término.
    - Es circular (solo repite la palabra) o no define realmente.

    Devuelve únicamente: correct o incorrect (en minúsculas, sin explicación).

    Palabra: {word}
    Definición de referencia: {correct_definition}
    Definición del modelo: {model_response}

    Respuesta:
    """
    
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    content = response.choices[0].message.content
    return content.strip().lower() if content else "incorrect"


def save_response(model: str, word: str, correct_definition: str, model_response: str, judgment: str = ""):
    """Save model response to output directory"""
    output_dir = Path(f"output/{model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    response_data = {
        "word": word,
        "correct_definition": correct_definition,
        "model_response": model_response,
        "judgment": judgment
    }
    
    file_path = output_dir / f"{word}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)


def load_response(model: str, word: str) -> dict:
    """Load existing response from output directory"""
    file_path = Path(f"output/{model}/{word}.json")
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def update_response_judgment(model: str, word: str, judgment: str):
    """Update existing response with judgment"""
    response_data = load_response(model, word)
    response_data["judgment"] = judgment
    
    output_dir = Path(f"output/{model}")
    file_path = output_dir / f"{word}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)


def calculate_accuracy(model: str, vocabulary: list[dict]) -> float:
    """Calculate accuracy percentage for a model"""
    correct_count = 0
    total_count = len(vocabulary)
    
    for entry in vocabulary:
        word = entry["word"]
        response_data = load_response(model, word)
        if response_data.get("judgment") == "correct":
            correct_count += 1
    
    return (correct_count / total_count) * 100 if total_count > 0 else 0


def generate_summary(models: list[str], vocabulary: list[dict]):
    """Generate summary.json and display results table"""
    summary = {}
    
    for model in models:
        accuracy = calculate_accuracy(model, vocabulary)
        summary[model] = accuracy
    
    # Save summary.json
    with open('summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Display results table
        console = Console()
        table = Table(title="Model Performance Summary")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Accuracy (%)", style="magenta", justify="right")
        table.add_column("Correct", style="green", justify="right")

        # Compute number of correct definitions per model
        correct_counts: dict[str, int] = {}
        for model in summary.keys():
            correct = 0
            for entry in vocabulary:
                word = entry["word"]
                response_data = load_response(model, word)
                if response_data.get("judgment") == "correct":
                    correct += 1
            correct_counts[model] = correct

        # Monkey-patch add_row so existing loop (adding only 2 args) appends the correct count
        original_add_row = table.add_row

        def patched_add_row(*args, **kwargs):
            # Expecting (model, accuracy_str)
            if len(args) == 2:
                model_name, accuracy_str = args
                return original_add_row(model_name, accuracy_str, str(correct_counts.get(model_name, "-")), **kwargs)
            return original_add_row(*args, **kwargs)

        table.add_row = patched_add_row
    
    for model, accuracy in summary.items():
        table.add_row(model, f"{accuracy:.1f}%")
    
    console.print(table)


def main():
    # Load data
    models = load_models()
    prompts = load_prompts()
    vocabulary = load_vocabulary()
    
    prompt_template = prompts["prompt_a"]
    
    console = Console()
    console.print(f"[bold green]Starting evaluation with {len(models)} models and {len(vocabulary)} words[/bold green]")
    
    # Step 2: Prompt models
    for model in models:
        console.print(f"[bold blue]Processing model: {model}[/bold blue]")
        
        for entry in tqdm(vocabulary, desc=f"Prompting {model}"):
            word = entry["word"]
            correct_definition = entry["answer"]
            
            # Skip if already processed
            if load_response(model, word):
                continue
                
            model_response = prompt_model(word, model, prompt_template)
            save_response(model, word, correct_definition, model_response)
    
    # Step 3: Judge responses
    console.print("[bold yellow]Starting judgment phase...[/bold yellow]")
    
    for model in models:
        console.print(f"[bold blue]Judging responses for: {model}[/bold blue]")
        
        for entry in tqdm(vocabulary, desc=f"Judging {model}"):
            word = entry["word"]
            correct_definition = entry["answer"]
            
            response_data = load_response(model, word)
            if not response_data or response_data.get("judgment"):
                continue
                
            model_response = response_data["model_response"]
            judgment = judge_response(word, correct_definition, model_response)
            update_response_judgment(model, word, judgment)
    
    # Step 4: Generate summary
    console.print("[bold green]Generating summary...[/bold green]")
    generate_summary(models, vocabulary)


if __name__ == "__main__":
    main()