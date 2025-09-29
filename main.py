
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


def judge_response_b(word: str, correct_definition: str, model_response: str) -> str:
    """Use GPT-5 to judge if the two-sentence response is correct"""
    client = OpenAI()  # Uses standard OpenAI API
    
    judge_prompt = f"""
    Evalúa si las dos frases proporcionadas cumplen con los criterios para la palabra indicada.

    Criterios para marcar correct:
    - La primera frase debe contener la palabra '{word}' y usarla correctamente en contexto.
    - La segunda frase no debe contener la palabra '{word}', pero debe estar relacionada con la primera.
    - Ambas frases deben ser coherentes y complementar el significado de la palabra según la definición de referencia.
    - El uso de la palabra debe ser consistente con el significado de referencia.

    Marca incorrect si:
    - La palabra no se usa correctamente en la primera frase.
    - La segunda frase no está relacionada con la primera.
    - El significado usado no coincide con la definición de referencia.
    - Las frases son incoherentes o no tienen sentido.
    - La palabra aparece en la segunda frase cuando no debería.

    Devuelve únicamente: correct o incorrect (en minúsculas, sin explicación).

    Palabra: {word}
    Definición de referencia: {correct_definition}
    Respuesta del modelo: {model_response}

    Respuesta:
    """
    
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    content = response.choices[0].message.content
    return content.strip().lower() if content else "incorrect"


def save_response(model: str, word: str, correct_definition: str, model_response_a: str = "", model_response_b: str = "", judgment_a: str = "", judgment_b: str = ""):
    """Save model response to output directory"""
    output_dir = Path(f"output/{model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing data if available
    file_path = output_dir / f"{word}.json"
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
    else:
        response_data = {
            "word": word,
            "correct_definition": correct_definition,
            "model_response_a": "",
            "model_response_b": "",
            "judgment_a": "",
            "judgment_b": ""
        }
    
    # Update only provided fields
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


def calculate_accuracy(model: str, vocabulary: list[dict], prompt_key: str = "a") -> float:
    """Calculate accuracy percentage for a model for a specific prompt"""
    correct_count = 0
    total_count = len(vocabulary)
    
    judgment_field = f"judgment_{prompt_key}"
    
    for entry in vocabulary:
        word = entry["word"]
        response_data = load_response(model, word)
        if response_data.get(judgment_field) == "correct":
            correct_count += 1
    
    return (correct_count / total_count) * 100 if total_count > 0 else 0


def generate_summary(models: list[str], vocabulary: list[dict]):
    """Generate summary.json and display results table"""
    summary = {}
    
    for model in models:
        accuracy_a = calculate_accuracy(model, vocabulary, "a")
        accuracy_b = calculate_accuracy(model, vocabulary, "b")
        summary[model] = {
            "prompt_a": accuracy_a,
            "prompt_b": accuracy_b
        }
    
    # Save summary.json
    with open('summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Display results table
    console = Console()
    table = Table(title="Model Performance Summary")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Prompt A (%)", style="magenta", justify="right")
    table.add_column("Correct A", style="green", justify="right")
    table.add_column("Prompt B (%)", style="yellow", justify="right")
    table.add_column("Correct B", style="green", justify="right")

    # Compute number of correct definitions per model for both prompts
    for model in summary.keys():
        correct_a = 0
        correct_b = 0
        for entry in vocabulary:
            word = entry["word"]
            response_data = load_response(model, word)
            if response_data.get("judgment_a") == "correct":
                correct_a += 1
            if response_data.get("judgment_b") == "correct":
                correct_b += 1
        
        accuracy_a = summary[model]["prompt_a"]
        accuracy_b = summary[model]["prompt_b"]
        table.add_row(
            model, 
            f"{accuracy_a:.1f}%", 
            str(correct_a),
            f"{accuracy_b:.1f}%",
            str(correct_b)
        )
    
    console.print(table)


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
            
            # Process prompt A if not already done
            if not response_data.get("model_response_a"):
                model_response_a = prompt_model(word, model, prompt_template_a)
                save_response(model, word, correct_definition, model_response_a=model_response_a)
            
            # Process prompt B if not already done
            if not response_data.get("model_response_b"):
                model_response_b = prompt_model(word, model, prompt_template_b)
                save_response(model, word, correct_definition, model_response_b=model_response_b)
    
    # Step 3: Judge responses
    console.print("[bold yellow]Starting judgment phase...[/bold yellow]")
    
    for model in models:
        console.print(f"[bold blue]Judging responses for: {model}[/bold blue]")
        
        for entry in tqdm(vocabulary, desc=f"Judging {model}"):
            word = entry["word"]
            correct_definition = entry["answer"]
            
            response_data = load_response(model, word)
            if not response_data:
                continue
            
            # Judge prompt A response if not already judged
            if response_data.get("model_response_a") and not response_data.get("judgment_a"):
                model_response_a = response_data["model_response_a"]
                judgment_a = judge_response(word, correct_definition, model_response_a)
                update_response_judgment(model, word, judgment_a=judgment_a)
            
            # Judge prompt B response if not already judged
            if response_data.get("model_response_b") and not response_data.get("judgment_b"):
                model_response_b = response_data["model_response_b"]
                judgment_b = judge_response_b(word, correct_definition, model_response_b)
                update_response_judgment(model, word, judgment_b=judgment_b)
    
    # Step 4: Generate summary
    console.print("[bold green]Generating summary...[/bold green]")
    generate_summary(models, vocabulary)


if __name__ == "__main__":
    main()