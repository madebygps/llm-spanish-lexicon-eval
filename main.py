import csv
from openai import OpenAI
import json
from typing import Optional
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="local")


class DefinitionJudgment(BaseModel):
    correct: bool
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

with open("definitions.json") as dj:
    dictionary_definition_data = json.load(dj)

dictionary_lookup = {
    item["word"]: item["definition"] for item in dictionary_definition_data
}


def get_llm_definition(word: str) -> str:
    llm_definition = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Dime la definición de la palabra {word}. Solo la definición y nada mas. Concreto y conciso.",
            }
        ],
        model="llama3.1:8b",
    )
    result = llm_definition.choices[0].message.content or ""
    return result.strip()


def llm_judges_itself(llm_definition: str, correct_definition: str, word: str) -> str:
    did_llm_provide_correct_definition = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"La definición correcta de {word} es {correct_definition}. "
                f"Comparando con tu defición: {llm_definition} dime si estuviste correcto o no. "
                "Response correcto o incorrecto y nada mas ",
            }
        ],
        model="llama3.1:8b",
    )
    result = did_llm_provide_correct_definition.choices[0].message.content or ""
    return result.strip()

def llm_judges_itself_bool(llm_definition: str, correct_definition: str, word: str) -> bool:
    """
    Structured output version: returns a boolean JSON `{"correct": true/false}`.
    Assumes the backend supports response_format with a JSON schema.
    """
    resp = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Evalúa si la definición propuesta coincide suficientemente con la definición correcta.\n"
                    f"Palabra: {word}\n"
                    f"Definición correcta: {correct_definition}\n"
                    f"Definición propuesta: {llm_definition}\n\n"
                    "Devuelve SOLO un JSON válido EXACTAMENTE de la forma:\n"
                    '{"correct": true}\n'
                    "o\n"
                    '{"correct": false}\n'
                    "Sin texto adicional."
                ),
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "definition_correctness",
                "schema": {
                    "type": "object",
                    "properties": {
                        "correct": {"type": "boolean"}
                    },
                    "required": ["correct"],
                    "additionalProperties": False,
                },
            },
        },
    )
    content = resp.choices[0].message.content or ""
    data = json.loads(content)  # simple parse; will raise if malformed
    return data["correct"]

def main():
    output_path = "results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as results_csv:
        writer = csv.writer(results_csv)
        writer.writerow(["Word", "LLM Definition", "Actual Definition", "LLM Judgment (RAW)", "Is Correct"])
        with open("words.txt", "r", encoding="utf-8") as f:
            for raw_word in (w for line in f for w in line.split()):
                word = raw_word.strip()
                if not word:
                    continue

                actual_definition = dictionary_lookup.get(word)
                if not actual_definition:
                    continue

                llm_definition = get_llm_definition(word)
                judgment_raw = llm_judges_itself(llm_definition, actual_definition, word)
                is_correct = llm_judges_itself_bool(llm_definition, actual_definition, word)

                writer.writerow([word, llm_definition, actual_definition, judgment_raw, is_correct])
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
