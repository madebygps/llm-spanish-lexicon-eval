from __future__ import annotations
from pathlib import Path
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator
from ollama import chat as ollama_chat
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).parent
SUITE_DIR = ROOT / "suite"
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_JUDGE_MODEL = "gpt-5"
ENV_OPENAI_KEY = "OPENAI_API_KEY"
ENV_JUDGE_MODEL = "JUDGE_MODEL"


class EvalRecord(BaseModel):
    """Single evaluation record for a (word, model) pair.

    judge_correct_a / judge_reasoning are populated after judging phase.
    """

    word: str
    definition: str
    model: str
    answer_a: Optional[str]
    judge_correct_a: Optional[bool] = None
    judge_reasoning: Optional[str] = None

    @field_validator("word", "definition", "model")
    @classmethod
    def _not_blank(cls, v: str) -> str: 
        if not v or not v.strip():
            raise ValueError("Value cannot be blank")
        return v.strip()

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)

class VocabEntry(BaseModel):
    word: str
    answer: str

    @field_validator("word", "answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:  # pragma: no cover
        if not v or not v.strip():
            raise ValueError("Vocabulary fields cannot be blank")
        return v.strip()


class Prompts(BaseModel):
    prompt_a: str = Field(..., description="Base prompt template for answer A")

    @field_validator("prompt_a")
    @classmethod
    def _validate_prompt(cls, v: str) -> str:  # pragma: no cover
        if "{word}" not in v:
            raise ValueError("prompt_a must contain '{word}' placeholder")
        return v


def load_suite() -> tuple[List[VocabEntry], Prompts, List[str]]:
    """Load & validate the evaluation suite.

    Returns:
        vocabulary: list of VocabEntry models
        prompts: validated Prompts model
        model_ids: list of model identifiers (raw strings)
    Raises:
        ValueError if any file content is structurally invalid.
    """
    with (SUITE_DIR / "vocabulary_short.json").open("r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
    if not isinstance(vocab_raw, list):  # pragma: no cover (defensive)
        raise ValueError("Vocabulary JSON must be a list of objects")
    vocabulary: List[VocabEntry] = [VocabEntry(**row) for row in vocab_raw]

    with (SUITE_DIR / "prompts.json").open("r", encoding="utf-8") as f:
        prompts_raw = json.load(f)
    if not isinstance(prompts_raw, dict):  # pragma: no cover
        raise ValueError("Prompts JSON must be an object")
    prompts = Prompts(**prompts_raw)

    with (SUITE_DIR / "models_list.txt").open("r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    model_ids = [ln.split()[0] for ln in raw]
    if not model_ids:
        raise ValueError("No model ids found in models_list.txt")
    return vocabulary, prompts, model_ids


def call_model(model: str, prompt: str) -> str:
    resp = ollama_chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()


def build_model_records(
    model: str, vocabulary: List[VocabEntry], prompts: Prompts
) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    for entry in tqdm(vocabulary, desc=f"Generating {model}"):
        word = entry.word
        definition = entry.answer
        prompt_a = prompts.prompt_a.format(word=word)
        try:
            resp = ollama_chat(
                model=model, messages=[{"role": "user", "content": prompt_a}]
            )
            ans_a = resp["message"]["content"].strip()
        except Exception as e:  # pragma: no cover
            ans_a = f"GENERATION_ERROR: {e}"
        records.append(
            EvalRecord(word=word, definition=definition, model=model, answer_a=ans_a)
        )
    return records


def judge_answer(
    openai_client, openai_model: str, word: str, definition: str, answer: str
) -> tuple[bool, str]:
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
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
        summary.append(
            {
                "model": model,
                "total": total,
                "correct": correct_a,
                "pct": round(pct_a, 2),
            }
        )
    summary.sort(key=lambda x: x["pct"], reverse=True)
    return summary


def write_jsonl(path: Path, records: Iterable[EvalRecord]) -> None:
    if not path.parent.exists():  # pragma: no cover
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def run_pipeline() -> int:
    load_dotenv()
    vocabulary, prompts, models = load_suite()
    all_records: List[EvalRecord] = []
    for model in models:
        per_model_path = OUTPUTS_DIR / model / f"result_{model.replace(':', '_')}.jsonl"
        per_model_records = build_model_records(
            model=model, vocabulary=vocabulary, prompts=prompts
        )
        write_jsonl(per_model_path, per_model_records)
        print(
            f"[OK] Wrote {len(per_model_records)} records for {model} -> {per_model_path}"
        )
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
            verdict, reasoning = judge_answer(
                client, judge_model_name, rec.word, rec.definition, rec.answer_a or ""
            )
            rec.judge_correct_a = verdict
            rec.judge_reasoning = reasoning
    write_jsonl(aggregated_path, all_records)
    for model in models:
        per_model_path = OUTPUTS_DIR / model / f"result_{model.replace(':', '_')}.jsonl"
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
