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
    def _not_blank(cls, v: str) -> str:
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

class SummaryRow(BaseModel):
    model: str
    total: int
    correct: int
    pct: float


def load_suite() -> tuple[List[VocabEntry], Prompts, List[str]]:
    """Load & validate the evaluation suite.

    Returns:
        vocabulary: list of VocabEntry models
        prompts: validated Prompts model
        model_ids: list of model identifiers (raw strings)
    Raises:
        ValueError if any file content is structurally invalid.
    """
    with (SUITE_DIR / "vocabulary_short.json").open(
        "r", encoding="utf-8"
    ) as vocab_file:
        vocab_data = json.load(vocab_file)
    if not isinstance(vocab_data, list):
        raise ValueError("Vocabulary JSON must be a list of objects")
    vocabulary: List[VocabEntry] = [VocabEntry(**row) for row in vocab_data]

    with (SUITE_DIR / "prompts.json").open("r", encoding="utf-8") as prompts_file:
        prompts_data = json.load(prompts_file)
    if not isinstance(prompts_data, dict):
        raise ValueError("Prompts JSON must be an object")
    prompts = Prompts(**prompts_data)

    with (SUITE_DIR / "models_list.txt").open("r", encoding="utf-8") as models_file:
        model_lines = [
            line.strip()
            for line in models_file
            if line.strip() and not line.strip().startswith("#")
        ]
    model_ids = [line.split()[0] for line in model_lines]

    return vocabulary, prompts, model_ids


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


def aggregate_summary(all_records: List[EvalRecord]) -> List[SummaryRow]:
    records_by_model: Dict[str, List[EvalRecord]] = {}
    for record in all_records:
        records_by_model.setdefault(record.model, []).append(record)
    summary_models: List[SummaryRow] = []
    for model_name, records_for_model in records_by_model.items():
        total_records = len(records_for_model)
        correct_answer_a = sum(1 for record in records_for_model if record.judge_correct_a)
        percent_correct_a = (correct_answer_a / total_records * 100) if total_records else 0.0
        summary_models.append(
            SummaryRow(
                model=model_name,
                total=total_records,
                correct=correct_answer_a,
                pct=round(percent_correct_a, 2),
            )
        )
    summary_models.sort(key=lambda row: row.pct, reverse=True)
    return summary_models


def write_jsonl(path: Path, records: Iterable[EvalRecord]) -> None:
    if not path.parent.exists():  # pragma: no cover
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def run_pipeline() -> int:
    """Run full pipeline (generate -> optional judge -> summary) with minimal ceremony.

    Keeps validation (Pydantic + load_suite) but strips extraneous branching/printing.
    """
    load_dotenv()
    vocabulary, prompts, models = load_suite()

    all_records: List[EvalRecord] = []
    for model in models:
        model_records = build_model_records(model=model, vocabulary=vocabulary, prompts=prompts)
        all_records.extend(model_records)
        write_jsonl(OUTPUTS_DIR / model / f"result_{model.replace(':','_')}.jsonl", model_records)

    aggregated_path = OUTPUTS_DIR / "detailed.jsonl"
    write_jsonl(aggregated_path, all_records)

    can_judge = OpenAI is not None and os.environ.get(ENV_OPENAI_KEY)
    if can_judge:
        client = OpenAI()  
        judge_model_name = os.environ.get(ENV_JUDGE_MODEL, DEFAULT_JUDGE_MODEL)
        for rec in tqdm(all_records, desc="Judging"):
            verdict, reasoning = judge_answer(client, judge_model_name, rec.word, rec.definition, rec.answer_a or "")
            rec.judge_correct_a = verdict
            rec.judge_reasoning = reasoning
        write_jsonl(aggregated_path, all_records)
        for model in models:
            write_jsonl(
                OUTPUTS_DIR / model / f"result_{model.replace(':','_')}.jsonl",
                (r for r in all_records if r.model == model),
            )
    else:
        if OpenAI is None:
            print("[WARN] openai not installed; skipping judging.", file=sys.stderr)
        elif not os.environ.get(ENV_OPENAI_KEY):
            print(f"[WARN] {ENV_OPENAI_KEY} not set; skipping judging.", file=sys.stderr)

    summary = aggregate_summary(all_records)
    summary_path = OUTPUTS_DIR / "summary.json"
    # Convert Pydantic models to plain dicts for JSON serialization
    summary_payload = [row.model_dump() for row in summary]
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    return 0


def main(_argv: Optional[List[str]] = None) -> int:
    return run_pipeline()


if __name__ == "__main__": 
    raise SystemExit(main())
