import json
from operator import itemgetter
from openai import OpenAI



def iterate_over_words(path="suite/vocabulary_complete.json"):
    with open(path, encoding="utf-8") as f:
        vocabulary_data = json.load(f)
    for word in vocabulary_data:
        ask_model_to_define_word(word)
        


def ask_model_to_define_word(word):
    """Test model's ability to choose correct Spanish definitions from multiple choices."""
    
    promptA = f"Dime la definición de la palabra '{word}'"
    promptb = f"Escribe dos frases, una con la palabra '{word}', y otra que no contenga esa palabra, pero que esté relacionada con la primera y complemente su significado."

    
    response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=50,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Lee la pregunta sobre definiciones en español y elige la respuesta correcta. "
                        "Responde solo con el número de la opción correcta."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{question}\n\n"
                        f"Opciones:\n{choices_text}\n\n"
                        f"Respuesta:"
                    ),
                },
            ],
        )

    model_output = response.choices[0].message.content.strip().lower()
    
    return model_output


def ask_model_to_use_word_in_sentences(word):
    """Test model's ability to use a word in contextually relevant sentences."""
    
    prompt = f"Escribe dos frases, una con la palabra '{word}', y otra que no contenga esa palabra, pero que esté relacionada con la primera y complemente su significado."

    response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Lee la instrucción y genera dos frases en español según lo solicitado."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        model_output = response.choices[0].message.content.strip()
        
      
    
    return model_output
def main():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    model = "mistral:latest"
    
    
    print("\n📚 Vocabulary Definitions Test:")
    answer= vocabulary_definitions(client=client, model=model)
    
    print(f"\nOverall Vocabulary Definitions Accuracy: {answer:.1%}")


if __name__ == "__main__":
    main()
