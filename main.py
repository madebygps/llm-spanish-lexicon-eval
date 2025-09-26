from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

try:
    from ollama import chat as ollama_chat
except Exception:  # pragma: no cover
    ollama_chat = None
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_, **__) -> bool:  # type: ignore
        return False
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore
        return it

ROOT = Path(__file__).parent
SUITE_DIR = ROOT / "suite"
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_JUDGE_MODEL = "gpt-5"
ENV_OPENAI_KEY = "OPENAI_API_KEY"
ENV_JUDGE_MODEL = "JUDGE_MODEL"

@dataclass
class EvalRecord:
    word: str
    definition: str
    model: str
    answer_a: Optional[str]
    judge_correct_a: Optional[bool] = None
    judge_reasoning: Optional[str] = None
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

def load_vocabulary() -> List[Dict[str, str]]:
    path = SUITE_DIR / "vocabulary_short.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_prompts() -> Dict[str, str]:
    path = SUITE_DIR / "prompts.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_model_ids() -> List[str]:
    path = SUITE_DIR / "models_list.txt"
    with path.open("r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return [ln.split()[0] for ln in raw]

def load_suite() -> tuple[List[Dict[str, str]], Dict[str, str], List[str]]:
    vocabulary = load_vocabulary()
    prompts = load_prompts()
    model_ids = load_model_ids()
    return vocabulary, prompts, model_ids

def ensure_output_dirs(models: Iterable[str], output_dir: Path = OUTPUTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        (output_dir / m).mkdir(parents=True, exist_ok=True)
def call_model(model: str, prompt: str) -> str:
    if ollama_chat is None:
        raise RuntimeError("ollama python package not available. Install dependencies or skip generation.")
    resp = ollama_chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()
def build_model_records(model: str, vocabulary: List[Dict[str, str]], prompts: Dict[str, str]) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    for entry in tqdm(vocabulary, desc=f"Generating {model}"):
        word = entry["word"]
        definition = entry["answer"]
        prompt_a = prompts["prompt_a"].format(word=word)
        try:
            if ollama_chat is None:
                raise RuntimeError("ollama python package not available.")
            resp = ollama_chat(model=model, messages=[{"role": "user", "content": prompt_a}])
            ans_a = resp["message"]["content"].strip()
        except Exception as e:  # pragma: no cover
            ans_a = f"GENERATION_ERROR: {e}"
        records.append(EvalRecord(word=word, definition=definition, model=model, answer_a=ans_a))
    return records

def judge_answer(openai_client, openai_model: str, word: str, definition: str, answer: str) -> tuple[bool, str]:
    system_prompt = (
        "Evalúa definiciones de vocabulario español. Responde primera línea YES o NO y segunda línea breve justificación. "
        "YES si captura el significado esencial sin contradicciones. NO si falla la categoría básica y rasgo distintivo, hay contradicción o atributo incompatible. Extra info correcta u omisiones menores de detalle no penalizan."
    )
    user_prompt = (
        f"PALABRA: {word}\n"
        f"DEFINICION_REFERENCIA: {definition}\n"
        f"RESPUESTA_MODELO:\n{answer.strip()}\n\n"
        "Evalúa ahora."
    )
    try:
        resp = openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:  # pragma: no cover
        return False, f"JUDGE_ERROR: {e}"
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    verdict_line = lines[0].upper() if lines else "NO"
    verdict = verdict_line.startswith("Y")
    reasoning = " ".join(lines[1:]) if len(lines) > 1 else ""
    return verdict, reasoning

def aggregate_summary(all_records: List[EvalRecord]) -> List[Dict[str, object]]:
    by_model: Dict[str, List[EvalRecord]] = {}
    for r in all_records:
        by_model.setdefault(r.model, []).append(r)
    summary = []
    for model, recs in by_model.items():
        total = len(recs)
        correct_a = sum(1 for r in recs if r.judge_correct_a)
        pct_a = (correct_a / total * 100) if total else 0.0
        summary.append({"model": model, "total": total, "correct": correct_a, "pct": round(pct_a, 2)})
    summary.sort(key=lambda x: x["pct"], reverse=True)
    return summary

def write_jsonl(path: Path, records: Iterable[EvalRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")

def run_pipeline() -> int:
    load_dotenv()
    vocabulary, prompts, models = load_suite()
    ensure_output_dirs(models)
    if ollama_chat is None:
        print("[ERROR] ollama package not installed; cannot generate.", file=sys.stderr)
        return 1
    all_records: List[EvalRecord] = []
    for model in models:
        per_model_path = OUTPUTS_DIR / model / f"result_{model.replace(':','_')}.jsonl"
        per_model_records = build_model_records(model=model, vocabulary=vocabulary, prompts=prompts)
        write_jsonl(per_model_path, per_model_records)
        print(f"[OK] Wrote {len(per_model_records)} records for {model} -> {per_model_path}")
        all_records.extend(per_model_records)
    aggregated_path = OUTPUTS_DIR / "detailed.jsonl"
    write_jsonl(aggregated_path, all_records)
    print(f"[OK] Aggregated detailed records -> {aggregated_path}")
    if OpenAI is None:
        print("[ERROR] openai package not installed; cannot judge.", file=sys.stderr)
        return 1
    if not os.environ.get(ENV_OPENAI_KEY):
        print(f"[ERROR] {ENV_OPENAI_KEY} not set; cannot judge.", file=sys.stderr)
        return 1
    client = OpenAI()
    judge_model_name = os.environ.get(ENV_JUDGE_MODEL, DEFAULT_JUDGE_MODEL)
    to_judge: List[EvalRecord] = [r for r in all_records if r.answer_a is not None]

    if to_judge:
        for rec in tqdm(to_judge, desc="Judging"):
            verdict, reasoning = judge_answer(client, judge_model_name, rec.word, rec.definition, rec.answer_a or "")
            rec.judge_correct_a = verdict
            rec.judge_reasoning = reasoning
    write_jsonl(aggregated_path, all_records)
    for model in models:
        per_model_path = OUTPUTS_DIR / model / f"result_{model.replace(':','_')}.jsonl"
        per_model_records = [rec for rec in all_records if rec.model == model]
        write_jsonl(per_model_path, per_model_records)
    print("[OK] Judging complete.")
    summary = aggregate_summary(all_records)
    summary_path = OUTPUTS_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Summary -> {summary_path}")
    return 0

def main(_argv: Optional[List[str]] = None) -> int:
    return run_pipeline()

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())