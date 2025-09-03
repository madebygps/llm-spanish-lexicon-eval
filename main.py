from openai import OpenAI
import requests
import json

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="local")

with open('definitions.json') as dj:
    dictionary_definition_data = json.load(dj)

dictionary_lookup = {item['word']: item['definition'] for item in dictionary_definition_data}

def main():
    with open("words.txt", "r") as f:
        for word in (w for line in f for w in line.split()):
            llm_definition = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Dime la definición de la palabra {word}. Solo la definición y nada mas. Concreto y conciso.",
                    }
                ],
                model="llama3.1:8b",
            )

            print("LLM DICE")
            print(f"{word}: {llm_definition.choices[0].message.content}")

            if word in dictionary_lookup:
                official_definition = dictionary_definition_data[word]

            print(f"DEFINICION CORRECT: {official_definition}")

            did_llm_provide_correct_definition = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"La definición correcta de {word} es {official_definition}. Comparando con tu defición: {llm_definition} dime si estuviste correcto o no. Response correcto o incorrecto y nada mas ",
                    }
                ],
                model="llama3.1:8b",
            )
            print(
                f"{word}: {did_llm_provide_correct_definition.choices[0].message.content}"
            )


if __name__ == "__main__":
    main()
