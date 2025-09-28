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


def get_model_filename(model: str) -> str:
    """Convert model name to safe filename format."""
    return model.replace(":", "_")


def setup_openai_client() -> tuple[OpenAI | None, str | None]:
    """Setup OpenAI client and judge model if available."""
    if OpenAI is None:
        print("[WARN] OpenAI not installed; skipping judging.", file=sys.stderr)
        return None, None
    if not os.environ.get(ENV_OPENAI_KEY):
        print(f"[WARN] {ENV_OPENAI_KEY} missing; skipping judging.", file=sys.stderr)
        return None, None
    return OpenAI(), os.environ.get(ENV_JUDGE_MODEL, DEFAULT_JUDGE_MODEL)

class PipelineResult(BaseModel):
    models: List[str]
    total_records: int
    judged_records: int
    summary: List[Dict[str, object]]
    outputs_dir: str


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


class VocabEntry(BaseModel):
    word: str
    answer: str


class Prompts(BaseModel):
    prompt_a: str = Field(..., description="Base prompt template for answer A")

    @field_validator("prompt_a")
    @classmethod
    def _validate_prompt(cls, v: str) -> str: 
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
        try:
            resp = ollama_chat(
                model=model, 
                messages=[{"role": "user", "content": prompts.prompt_a.format(word=entry.word)}]
            )
            answer_a = resp["message"]["content"].strip()
        except Exception as e:  # pragma: no cover
            answer_a = f"GENERATION_ERROR: {e}"
        
        records.append(EvalRecord(
            word=entry.word, 
            definition=entry.answer, 
            model=model, 
            answer_a=answer_a
        ))
    return records


def judge_answer(
    openai_client, openai_model: str, word: str, definition: str, answer: str
) -> tuple[bool, str]:
    try:
        resp = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": (
                    "Evalúa definiciones de vocabulario español. Responde primera línea YES o NO y segunda línea breve justificación. "
                    "YES si captura el significado esencial sin contradicciones. NO si falla la categoría básica y rasgo distintivo, hay contradicción o atributo incompatible. Extra info correcta u omisiones menores de detalle no penalizan."
                )},
                {"role": "user", "content": (
                    f"PALABRA: {word}\n"
                    f"DEFINICION_REFERENCIA: {definition}\n"
                    f"RESPUESTA_MODELO:\n{answer.strip()}\n\n"
                    "Evalúa ahora."
                )},
            ],
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:  # pragma: no cover
        return False, f"JUDGE_ERROR: {e}"
    
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    verdict = lines[0].upper().startswith("Y") if lines else False
    reasoning = " ".join(lines[1:]) if len(lines) > 1 else ""
    return verdict, reasoning


def aggregate_summary(all_records: List[EvalRecord]) -> List[Dict[str, object]]:
    records_by_model: Dict[str, List[EvalRecord]] = {}
    for record in all_records:
        records_by_model.setdefault(record.model, []).append(record)
    summary: List[Dict[str, object]] = []
    for model_name, model_records in records_by_model.items():
        total = len(model_records)
        correct = sum(1 for r in model_records if r.judge_correct_a)
        pct = round((correct / total * 100) if total else 0.0, 2)
        summary.append(
            {"model": model_name, "total": total, "correct": correct, "pct": pct}
        )
    summary.sort(key=lambda r: float(r["pct"]), reverse=True)  # type: ignore[arg-type]
    return summary


def write_jsonl(path: Path, records: Iterable[EvalRecord], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")


def run_pipeline() -> PipelineResult:
    load_dotenv()
    vocabulary, prompts, models = load_suite()

    # Generate all model responses
    all_records: List[EvalRecord] = []
    for model in models:
        all_records.extend(build_model_records(model, vocabulary, prompts))

    # Judge responses if OpenAI is available
    client, judge_model = setup_openai_client()
    judged = 0
    if client and judge_model:
        for rec in tqdm(all_records, desc="Judging"):
            rec.judge_correct_a, rec.judge_reasoning = judge_answer(
                client, judge_model, rec.word, rec.definition, rec.answer_a or ""
            )
            judged += 1

    # Write all output files once
    for model in models:
        model_filename = get_model_filename(model)
        write_jsonl(
            OUTPUTS_DIR / model / f"result_{model_filename}.jsonl",
            (r for r in all_records if r.model == model),
        )
    
    write_jsonl(OUTPUTS_DIR / "detailed.jsonl", all_records)
    summary = aggregate_summary(all_records)
    with (OUTPUTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return PipelineResult(
        models=models,
        total_records=len(all_records),
        judged_records=judged,
        summary=summary,
        outputs_dir=str(OUTPUTS_DIR),
    )


def main(_argv: Optional[List[str]] = None) -> int:
    _res = run_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
