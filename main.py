import json
from operator import itemgetter
from openai import OpenAI


def cloze_mcq(client, path="suite/cloze_mcq.jsonl", model="mistral:latest"):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total = 0
    correct = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task, question, choices, correct_answer = itemgetter(
                "task", "question", "choices", "answer"
            )(data)

            # Label choices A), B), C)...
            labeled = [f"{letters[i]}) {opt}" for i, opt in enumerate(choices)]
            letter_map = {letters[i]: choices[i] for i in range(len(choices))}

            # Ask the model
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=5,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente que responde preguntas de opción múltiple en español. "
                            "Responde SOLO con la letra de la opción correcta (A, B, C, ...). "
                            "No expliques tu razonamiento."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Pregunta: {question}\n"
                            "Opciones:\n" + "\n".join(labeled) + "\n"
                            "Respuesta (solo la letra):"
                        ),
                    },
                ],
            )

            # Raw model output
            model_output = response.choices[0].message.content.strip()

            # Normalize: letter → choice text
            letter = None
            if model_output and len(model_output) >= 1:
                first_char = model_output[0].upper()
                if first_char in letter_map:
                    letter = first_char

            normalized_output = (letter_map[letter] if letter else model_output).strip().lower()

            # Normalize correct answer
            correct_answer_clean = correct_answer.strip().lower()

            is_right = normalized_output == correct_answer_clean
            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Model said: {model_output}\n"
                f"  Interpreted as: {normalized_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )

    print(f"Final Accuracy: {correct}/{total} = {correct / total:.1%}")


def everyday_qa(client, path="suite/everyday_qa.jsonl", model="mistral:latest"):
    total = 0
    correct = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task, question, correct_answer = itemgetter("task", "question", "answer")(
                data
            )

            # Ask the model
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=20,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente que responde preguntas en español. "
                            "Responde solo con la respuesta correcta, sin explicaciones."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Pregunta: {question}",
                    },
                ],
            )

            # Raw model output
            model_output = response.choices[0].message.content.strip()

            # Normalize
            normalized_output = model_output.lower()
            correct_answer_clean = correct_answer.strip().lower()

            is_right = normalized_output == correct_answer_clean
            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Model said: {model_output}\n"
                f"  Interpreted as: {normalized_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )

    print(f"Final Accuracy: {correct}/{total} = {correct / total:.1%}")


def mcq_reasoning(client, path="suite/mcq_reasoning.jsonl", model="mistral:latest"):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total = 0
    correct = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task, context, question, choices, correct_answer = itemgetter(
                "task", "context", "question", "choices", "answer"
            )(data)

            # Label choices A), B), C)...
            labeled = [f"{letters[i]}) {opt}" for i, opt in enumerate(choices)]
            letter_map = {letters[i]: choices[i] for i in range(len(choices))}

            # Ask the model
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=5,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente que responde preguntas de opción múltiple en español. "
                            "Responde SOLO con la letra de la opción correcta (A, B, C, ...). "
                            "No expliques tu razonamiento."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Contexto: {context}\n"
                            f"Pregunta: {question}\n"
                            "Opciones:\n" + "\n".join(labeled) + "\n"
                            "Respuesta (solo la letra):"
                        ),
                    },
                ],
            )

            # Raw model output
            model_output = response.choices[0].message.content.strip()

            # Normalize: convert letter → choice text if possible
            letter = None
            if model_output and len(model_output) >= 1:
                first_char = model_output[0].upper()
                if first_char in letter_map:
                    letter = first_char

            normalized_output = (letter_map[letter] if letter else model_output).strip().lower()

            # Normalize correct answer
            correct_answer_clean = correct_answer.strip().lower()

            is_right = normalized_output == correct_answer_clean
            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Model said: {model_output}\n"
                f"  Interpreted as: {normalized_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )

    print(f"Final Accuracy: {correct}/{total} = {correct / total:.1%}")


def reading_comprehension(
    client, path="suite/reading_comprehension.jsonl", model="mistral:latest"
):
    total = 0
    correct = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task, context, question, correct_answer = itemgetter(
                "task", "context", "question", "answer"
            )(data)

            # Ask the model
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=100,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente que responde preguntas de comprensión lectora en español. "
                            "Responde solo con la respuesta correcta, sin explicaciones."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (f"Contexto: {context}\nPregunta: {question}"),
                    },
                ],
            )

            # Raw model output
            model_output = response.choices[0].message.content.strip()

            # Normalize
            normalized_output = model_output.lower()
            correct_answer_clean = correct_answer.strip().lower()

            is_right = normalized_output == correct_answer_clean
            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Model said: {model_output}\n"
                f"  Interpreted as: {normalized_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )

    print(f"Final Accuracy: {correct}/{total} = {correct / total:.1%}")


def main():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    model = "phi3:3.8b"
    mcq_reasoning(client=client, model=model)
    reading_comprehension(client=client, model=model)
    everyday_qa(client=client, model=model)
    cloze_mcq(client=client, model=model)


if __name__ == "__main__":
    main()
