## LLM Spanish Lexicon Eval

Evaluate how well local LLMs (via Ollama) handle Spanish vocabulary definitions and contextual sentence usage. Answers are judged automatically by an OpenAI model to compute accuracy metrics for two prompt styles (A: definition, B: two related sentences).

### Requirements

1. Python 3.13+
2. [uv](https://github.com/astral-sh/uv) for dependency + virtualenv management
3. Local models pulled into Ollama (e.g. `ollama pull mistral:latest`)
4. OpenAI API key in a `.env` file at repository root:
```
OPENAI_API_KEY=sk-...
```

### Install deps

```bash
uv sync  # or: uv add ollama python-dotenv tqdm openai
```

### Run a quick smoke test

Dry run (no model calls):
```bash
uv run python main.py --dry-run --limit 5
```

Generate only (skip judging):
```bash
uv run python main.py --limit 10 --skip-judge
```

Judge existing generation (resume):
```bash
uv run python main.py --skip-generation --resume
```

Full evaluation (all models & words):
```bash
uv run python main.py
```

Limit vocab items for faster iteration:
```bash
uv run python main.py --limit 25 --models mistral:latest llama3.1:latest
```

### Outputs

```
outputs/
	detailed.jsonl         # All (word, model) records
	summary.json           # Per-model accuracy percentages
	<model>/result_<model>.jsonl  # Per-model records
```

Each JSONL record structure:
```jsonc
{
	"word": "acelguilla",
	"definition": "... ground truth ...",
	"model": "mistral:latest",
	"answer_a": "...",
	"answer_b": "...",
	"judge_correct_a": true,
	"judge_correct_b": false,
	"judge_reasoning": "YES: ... | NO: ..."
}
```

### CLI Options

Run `uv run python main.py --help` for the latest list. Key flags:

* `--limit N` limit vocabulary items
* `--models m1 m2` subset of models
* `--skip-generation` skip local model calls
* `--skip-judge` skip OpenAI judging
* `--resume` reuse existing JSONL files (append missing judgments)
* `--openai-model MODEL` change judge model (default: gpt-4o-mini)
* `--gen-concurrency N` parallel requests per model during generation (default 1)

### Concurrency & Ollama Tuning

This project now supports parallel requests *within a single model* (models still handled one after another) via `--gen-concurrency`.

Example (4 parallel in-flight generations per model):
```bash
uv run python main.py --gen-concurrency 4 --limit 50 --models mistral:latest
```

Server side, adjust Ollama (before starting `ollama serve` or via `~/.ollama/config.json`):
```json
{
	"num_parallel": 4,
	"max_loaded_models": 1
}
```
or environment variables if manually launching:
```bash
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve
```

Guidelines:
* Keep `max_loaded_models=1` initially to avoid memory churn.
* Increase `num_parallel` (server) to be ≥ `--gen-concurrency` (client) for true concurrency; otherwise excess requests queue.
* Monitor memory; parallelism expands effective context memory footprint.

If you don't control server startup (daemon already running) edit the config file then restart the service for changes to apply.

### Judging Criteria (Summary)

Prompt A: Response must faithfully capture essential meaning of the reference definition (no major omissions or hallucinations).

Prompt B: Two coherent sentences — one must use the target word correctly; the other related sentence should complement meaning and stay consistent with the reference.

Judge outputs YES / NO for each variant.

### Troubleshooting

* Missing Ollama python package: ensure dependencies installed (`uv sync`).
* Model not found: pull it with `ollama pull <model>`.
* No OpenAI key: create `.env` with OPENAI_API_KEY.

### Next Ideas

* Parallel generation per model (async) for speed.
* Add caching layer for previously seen (model, prompt) pairs.
* Precision / recall style grading with more granular rubric.

---
Happy evaluating! ✨
