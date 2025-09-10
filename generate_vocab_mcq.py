#!/usr/bin/env python3
"""
Script to generate MCQ vocabulary tests from all.json
Creates plausible distractors for each word definition.
"""
import json
import random

def generate_distractors(word, correct_definition, all_definitions):
    """Generate 3 plausible distractors for a given word definition."""
    
    # Categories of distractors based on word patterns and meanings
    distractor_templates = {
        # For plants/animals
        'biological': [
            "Planta herbácea que crece en terrenos húmedos y se utiliza tradicionalmente en medicina popular.",
            "Animal doméstico de pequeño tamaño que se cría por su carne y sus productos derivados.",
            "Insecto volador que se alimenta del néctar de las flores y contribuye a la polinización.",
            "Árbol frutal de origen mediterráneo que se cultiva por sus frutos comestibles y aceite."
        ],
        # For tools/objects
        'tools': [
            "Herramienta manual utilizada en trabajos de carpintería y construcción para cortar madera.",
            "Instrumento mecánico empleado para medir distancias y ángulos con gran precisión.",
            "Dispositivo electrónico que permite la comunicación a larga distancia mediante ondas.",
            "Utensilio de cocina fabricado en metal que se usa para preparar y servir alimentos."
        ],
        # For people/professions
        'people': [
            "Persona especializada en el cuidado y tratamiento de enfermedades en animales domésticos.",
            "Profesional que se dedica a la enseñanza de materias académicas en instituciones educativas.",
            "Trabajador especializado en la reparación y mantenimiento de equipos electrónicos.",
            "Funcionario encargado de hacer cumplir las leyes y mantener el orden público."
        ],
        # For actions/verbs
        'actions': [
            "Realizar una acción repetitiva con el objetivo de mejorar una habilidad específica.",
            "Organizar elementos de manera sistemática para facilitar su uso posterior.",
            "Transmitir información importante a otras personas mediante diferentes medios.",
            "Examinar cuidadosamente algo para determinar su estado o funcionamiento."
        ],
        # For concepts/abstract
        'concepts': [
            "Conjunto de normas y principios que regulan el comportamiento en una sociedad.",
            "Proceso mental mediante el cual se adquieren y procesan nuevos conocimientos.",
            "Sistema de creencias y valores compartidos por un grupo de personas.",
            "Método científico utilizado para investigar y comprender fenómenos naturales."
        ],
        # For places/locations
        'places': [
            "Establecimiento comercial donde se venden productos alimenticios y artículos de primera necesidad.",
            "Edificio público destinado a actividades culturales, educativas y de entretenimiento.",
            "Zona geográfica caracterizada por condiciones climáticas y paisajes específicos.",
            "Construcción arquitectónica destinada al culto religioso y ceremonias espirituales."
        ]
    }
    
    # Classify the word/definition to choose appropriate distractors
    definition_lower = correct_definition.lower()
    
    if any(word in definition_lower for word in ['planta', 'árbol', 'animal', 'mamífero', 'ave', 'pez', 'insecto']):
        category = 'biological'
    elif any(word in definition_lower for word in ['herramienta', 'instrumento', 'dispositivo', 'aparato', 'máquina']):
        category = 'tools'
    elif any(word in definition_lower for word in ['persona', 'profesional', 'trabajador', 'especialista']):
        category = 'people'
    elif any(word in definition_lower for word in ['acción', 'hacer', 'realizar', 'proceso']):
        category = 'actions'
    elif any(word in definition_lower for word in ['sala', 'edificio', 'lugar', 'establecimiento']):
        category = 'places'
    else:
        category = 'concepts'
    
    # Get base distractors from template
    base_distractors = random.sample(distractor_templates[category], 3)
    
    # Add some variety by mixing in other categories occasionally
    if random.random() < 0.3:  # 30% chance to mix categories
        other_categories = [k for k in distractor_templates.keys() if k != category]
        mixed_category = random.choice(other_categories)
        base_distractors[-1] = random.choice(distractor_templates[mixed_category])
    
    return base_distractors

def create_vocabulary_mcq():
    """Create comprehensive vocabulary MCQ from all.json"""
    
    # Load all definitions
    with open('all.json', 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Load existing vocabulary (to preserve the good manual ones)
    try:
        with open('suite/vocabulary.json', 'r', encoding='utf-8') as f:
            existing_vocab = json.load(f)
        existing_words = {item['word'] for item in existing_vocab}
    except:
        existing_vocab = []
        existing_words = set()
    
    # Create MCQ format for all words
    vocab_mcq = list(existing_vocab)  # Start with existing good ones
    
    for item in all_data:
        word = item['word']
        definition = item['definition']
        
        # Skip if we already have a good manual version
        if word in existing_words:
            continue
        
        # Clean up definition
        clean_definition = definition.replace('1. f. ', '').replace('1. m. ', '').replace('1. adj. ', '').replace('1. tr. ', '').replace('1. intr. ', '')
        clean_definition = clean_definition.split('Sin.:')[0].split('Ant.:')[0].strip()
        
        # Generate distractors
        distractors = generate_distractors(word, clean_definition, all_data)
        
        # Create choices (correct + 3 distractors)
        choices = [clean_definition] + distractors
        random.shuffle(choices)
        
        vocab_item = {
            "word": word,
            "question": f"¿Cuál es la definición correcta de '{word}'?",
            "choices": choices,
            "answer": clean_definition
        }
        
        vocab_mcq.append(vocab_item)
    
    # Save to file
    with open('suite/vocabulary_complete.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_mcq, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(vocab_mcq)} vocabulary MCQ items")
    print("Saved to suite/vocabulary_complete.json")

if __name__ == "__main__":
    create_vocabulary_mcq()