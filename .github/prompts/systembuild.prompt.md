---
mode: agent
---

# Overview

I [read a paper](https://arxiv.org/abs/2403.15491) recently where the authors hypothesized that open-weight LLMs don’t understand most Spanish words. To test this, they randomly selected 100 words from Diccionario del Español Actual and queried 12 popular open conversational models. This was done in 2024 so I would like to see if now in 2025 models are any better at this.

Here’s are the prompts used:

```json
{
  "prompt_a": "Dime la definición de la palabra '{word}'.",
  "prompt_b": "Escribe dos frases, una con la palabra '{word}', y otra que no contenga esa palabra, pero que esté relacionada con la primera y complemente su significado."
}
```

and here is a list of models they tested:


LLaMA-2-7B (Meta)
LLaMA-2-13B (Meta)
LLaMA-2-70B (Meta)
Mistral-7B (Mistral AI)
Mixtral-46.7B (Mistral AI, mixture-of-experts)
Solar-10.7B (scaling variant of Mistral-7B)
Yi-6B (01.AI, English + Chinese optimized)
Yi-34B (01.AI, English + Chinese optimized)
Gemma-7B (Google, recently released at time of paper)
Bloomz-7B1.5 (multilingual, BigScience)
Flor-6.3B-Instructed (Spanish-optimized, derived from Bloom)
Bertin-6B (Spanish-tuned GPT-J 6B)

and here is a list of words they used: 

acelguilla, agüista, antidiarreico, apealar, apparat, arante, ardilla, bátavo, bicicross, bifocal, cantautor, cantinero, cartulina, centavo, cerebrotónico, chalaza, chigüire, coliseo, conspirar, corbata, corralero, cosmotrón, crístico, cuentapartícipe, dabuti, dactiloscopia, deformabilidad, desinfección, desinsectación, desmitificador, diligentemente, emparejador, empurpurado, epifito, escorar, estadista, esteatosis, estuco, exequias, faisánido, fétido, floración, fotogramétrico, funcionario, gabato, garcilla, giroscopio, helenizante, hipogino, incrustar, intercadencia, jaín, lipotimia, magnolia, manes, meano, mediar, mensualizar, mesmerización, mestizo, minuto, mochalero, modisto, monásticamente, morra, nefólogo, novatada, ovni, pagar, paleteo, palmítico, paralogismo, pasarratos, perrillo, pezuña, pinabete, pitahaya, postrar, prédica, prolongador, provinciano, puzzle, quepis, raor, reciclado, rememorar, ridiculización, sagum, salbanda, socialmente, speed, standarizar, sublimado, superbomba, talgo, tornajo, treinta, tundidor, vega, verraquear.



# Task

Your task is to help me build a system that automates this evaluation.

1. in `main.py` CREATE 



models to test: 

Llama 3.1 8B Instruct – strong small baseline
ollama pull llama3.1:8b-instruct

Llama 3.1 70B Instruct – “big model” contrast to show size effects
ollama pull llama3.1:70b-instruct
Tip: use a quantized tag (e.g., :70b-instruct-q4_K_M) if VRAM/latency is tight.

Gemma 2 9B – different lineage (Google), good multilingual-ish baseline
ollama pull gemma2:9b-instruct

Qwen3 8B Instruct – very good multilingual family; interesting for Spanish lexicon
ollama pull qwen3:8b-instruct

Mixtral 8x7B Instruct – MoE architecture; nice comparison vs. dense models
ollama pull mixtral:8x7b-instruct