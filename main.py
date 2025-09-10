import json
from operator import itemgetter
from openai import OpenAI


def cloze_mcq(client, path="suite/cloze_mcq.jsonl", model="mistral:latest"):
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

            # Simple numbered list - no letters
            choices_text = "\n".join(choices)

            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=10,  # Severely limit tokens to force concise answers
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "Responde ÚNICAMENTE con la opción correcta. "
                            "NO expliques. NO escribas oraciones completas. "
                            "Solo escribe la respuesta exacta de las opciones dadas."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"{question}\n\nOpciones:\n{choices_text}\n\nRespuesta:"
                    },
                ],
            )

            model_output = response.choices[0].message.content.strip().lower()
            correct_answer_clean = correct_answer.strip().lower()

            # More flexible matching for direct choice approach
            is_right = (
                model_output == correct_answer_clean or 
                correct_answer_clean in model_output or
                # Check if any choice matches and appears in model output
                any(choice.strip().lower() in model_output for choice in choices 
                    if choice.strip().lower() == correct_answer_clean)
            )

            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Model said: {model_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )

    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy

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

            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=20,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Responde preguntas en español con respuestas cortas y precisas. "
                            "Solo da la respuesta correcta, sin explicaciones."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{question}",  # Removed "Pregunta:" prefix for cleaner prompt
                    },
                ],
            )

            model_output = response.choices[0].message.content.strip()
            normalized_output = model_output.lower()
            correct_answer_clean = correct_answer.strip().lower()

            # More flexible matching for comprehension testing
            is_right = (
                normalized_output == correct_answer_clean or 
                correct_answer_clean in normalized_output or
                # Handle common variations (articles, punctuation)
                normalized_output.replace('.', '').replace(',', '') == correct_answer_clean
            )

            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Model said: {model_output}\n"
                f"  Interpreted as: {normalized_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )

    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy

def mcq_reasoning(client, path="suite/mcq_reasoning.jsonl", model="mistral:latest"):
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

            # Simple list without letters - just the choices
            choices_text = "\n".join(choices)

            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=50,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Lee el contexto y responde la pregunta eligiendo la mejor opción. "
                            "Responde solo con la opción correcta, sin explicaciones."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Contexto: {context}\n"
                            f"Pregunta: {question}\n"
                            f"Opciones:\n{choices_text}\n"
                            f"Respuesta:"
                        ),
                    },
                ],
            )

            model_output = response.choices[0].message.content.strip().lower()
            correct_answer_clean = correct_answer.strip().lower()

            # Flexible matching for comprehension testing
            is_right = (
                model_output == correct_answer_clean or 
                correct_answer_clean in model_output or
                # Check if any choice matches and appears in model output
                any(choice.strip().lower() in model_output for choice in choices 
                    if choice.strip().lower() == correct_answer_clean)
            )

            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Context: {context[:50]}...\n"
                f"  Model said: {model_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )

    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy

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

            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=100,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Lee el contexto y responde la pregunta basándote en la información proporcionada. "
                            "Responde de forma concisa y precisa."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (f"Contexto: {context}\n\nPregunta: {question}"),
                    },
                ],
            )

            model_output = response.choices[0].message.content.strip()
            normalized_output = model_output.lower()
            correct_answer_clean = correct_answer.strip().lower()

            # More flexible matching for reading comprehension
            is_right = (
                normalized_output == correct_answer_clean or 
                correct_answer_clean in normalized_output or
                # Handle punctuation variations
                normalized_output.replace('.', '').replace(',', '').strip() == correct_answer_clean or
                # Check if key words match (for longer answers)
                all(word in normalized_output for word in correct_answer_clean.split() if len(word) > 2)
            )

            total += 1
            correct += int(is_right)

            print(
                f"Q{total}: {question}\n"
                f"  Context: {context[:100]}...\n"
                f"  Model said: {model_output}\n"
                f"  Correct answer: {correct_answer_clean}\n"
                f"  Result: {'✅ Correct' if is_right else '❌ Wrong'}\n"
            )
            
    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy

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
    
    print(f"🇪🇸 Spanish Comprehension Evaluation for {model}")
    print("=" * 50)
    
    # Run all tests and collect scores
    print("\n📝 MCQ Reasoning Test:")
    mcq_score = mcq_reasoning(client=client, model=model)
    
    print("\n📖 Reading Comprehension Test:")
    reading_score = reading_comprehension(client=client, model=model)
    
    print("\n❓ Everyday Q&A Test:")
    qa_score = everyday_qa(client=client, model=model)
    
    print("\n✏️ Cloze MCQ Test:")
    cloze_score = cloze_mcq(client=client, model=model)
    
    print("\n📚 Vocabulary Definitions Test:")
    vocab_score = vocabulary_definitions(client=client, model=model)
    
    # Calculate weighted final grade
    weights = {
        'MCQ Reasoning': 0.25,
        'Reading Comprehension': 0.25, 
        'Everyday Q&A': 0.20,
        'Cloze MCQ': 0.15,
        'Vocabulary Definitions': 0.15
    }
    
    final_grade = (
        mcq_score * weights['MCQ Reasoning'] +
        reading_score * weights['Reading Comprehension'] +
        qa_score * weights['Everyday Q&A'] +
        cloze_score * weights['Cloze MCQ'] +
        vocab_score * weights['Vocabulary Definitions']
    )
    
    # Display final results
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    print(f"MCQ Reasoning:         {mcq_score:.1%} (weight: {weights['MCQ Reasoning']:.0%})")
    print(f"Reading Comprehension: {reading_score:.1%} (weight: {weights['Reading Comprehension']:.0%})")
    print(f"Everyday Q&A:          {qa_score:.1%} (weight: {weights['Everyday Q&A']:.0%})")
    print(f"Cloze MCQ:             {cloze_score:.1%} (weight: {weights['Cloze MCQ']:.0%})")
    print(f"Vocabulary Definitions: {vocab_score:.1%} (weight: {weights['Vocabulary Definitions']:.0%})")
    print("-" * 50)
    print(f"🎯 OVERALL GRADE:       {final_grade:.1%}")
    
    # Grade interpretation
    if final_grade >= 0.9:
        grade_letter = "A+"
        interpretation = "Excellent Spanish comprehension"
    elif final_grade >= 0.8:
        grade_letter = "A"
        interpretation = "Very good Spanish comprehension"
    elif final_grade >= 0.7:
        grade_letter = "B"
        interpretation = "Good Spanish comprehension"
    elif final_grade >= 0.6:
        grade_letter = "C"
        interpretation = "Fair Spanish comprehension"
    elif final_grade >= 0.5:
        grade_letter = "D"
        interpretation = "Basic Spanish comprehension"
    else:
        grade_letter = "F"
        interpretation = "Poor Spanish comprehension"
    
    print(f"📈 Grade: {grade_letter} - {interpretation}")
    print("=" * 50)


if __name__ == "__main__":
    main()
