"""Client for interacting with AI models."""

from openai import OpenAI


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
    """Use GPT-5 to judge if the model response for prompt B demonstrates understanding of the word"""
    client = OpenAI()  # Uses standard OpenAI API
    
    judge_prompt = f"""
    Evalúa si las dos frases proporcionadas demuestran una comprensión correcta de la palabra indicada.

    Criterios para marcar correct:
    - La primera frase usa la palabra '{word}' de manera apropiada y coherente con su definición.
    - La segunda frase está relacionada con la primera y complementa el significado sin usar la palabra.
    - Ambas frases juntas revelan comprensión del significado de la palabra.
    - El uso contextual de la palabra es correcto según su definición de referencia.

    Marca incorrect si:
    - La palabra se usa incorrectamente en la primera frase.
    - Las frases no están relacionadas o no complementan el significado.
    - La segunda frase usa la palabra cuando no debería.
    - Las frases no demuestran comprensión real del significado de la palabra.
    - El contexto de uso contradice la definición de referencia.

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