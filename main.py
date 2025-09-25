import json
from operator import itemgetter
from openai import OpenAI


def vocabulary_definitions(client, path="suite/vocabulary_complete.json", model="mistral:latest"):
    """Test model's ability to choose correct Spanish definitions from multiple choices."""
    import random
    
    total = 0
    correct = 0

    # Load all vocabulary questions
    with open(path, encoding="utf-8") as f:
        vocabulary_data = json.load(f)
    
    # Take a random sample for testing (20 questions is reasonable)
    sample_size = min(20, len(vocabulary_data))
    test_items = random.sample(vocabulary_data, sample_size)

    for item in test_items:
        word = item["word"]
        question = item["question"]
        choices = item["choices"].copy()  # Make a copy to avoid modifying original
        correct_answer = item["answer"]
        
        # Shuffle choices to randomize correct answer position
        random.shuffle(choices)
        
        # Format choices
        choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])

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
        correct_answer_clean = correct_answer.strip().lower()

        # Check if model's response matches the correct answer
        is_right = False
        
        # Handle numbered responses (e.g., "1", "2", "3", "4")
        if model_output.strip().isdigit():
            choice_num = int(model_output.strip())
            if 1 <= choice_num <= len(choices):
                selected_choice = choices[choice_num - 1].strip().lower()
                if selected_choice == correct_answer_clean:
                    is_right = True
        
        # Handle responses that start with a number (e.g., "1.", "2)", "1 -")
        else:
            first_word = model_output.split()[0] if model_output.split() else ""
            # Extract number from patterns like "1.", "2)", "3-", etc.
            import re
            number_match = re.match(r'^(\d+)', first_word)
            if number_match:
                choice_num = int(number_match.group(1))
                if 1 <= choice_num <= len(choices):
                    selected_choice = choices[choice_num - 1].strip().lower()
                    if selected_choice == correct_answer_clean:
                        is_right = True
        
        # Find which position the correct answer ended up in after shuffling
        correct_position = -1
        for i, choice in enumerate(choices):
            if choice.strip().lower() == correct_answer_clean:
                correct_position = i + 1
                break

        total += 1
        correct += int(is_right)

        print(
            f"Q{total}: {word} - {question}\n"
            f"  Model said: {model_output}\n"
            f"  Correct answer was option #{correct_position}: {correct_answer_clean[:60]}...\n"
            f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
        )

    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy

def main():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    model = "mistral:latest"
    
    
    print("\n📚 Vocabulary Definitions Test:")
    vocab_score = vocabulary_definitions(client=client, model=model)
    
    print(f"\nOverall Vocabulary Definitions Accuracy: {vocab_score:.1%}")


if __name__ == "__main__":
    main()
