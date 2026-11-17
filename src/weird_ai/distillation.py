
"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
import math
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)

REQUIRED_FIELDS = ("prompt", "teacher_lyrics", "teacher_notes", "quality_score")
TEXT_FIELDS = ("prompt", "teacher_lyrics", "teacher_notes")

def validate_teacher_example(example: dict) -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """
    # TODO:
    # 1. Check that all required fields are present.
    if not isinstance(example, dict):
            return False
    
    if any(key not in example for key in REQUIRED_FIELDS):
        return False
    
    # 2. Check that prompt, teacher_lyrics, and teacher_notes are non-empty strings.
    for key in TEXT_FIELDS:
            value = example[key]
            if not isinstance(value, str) or not value.strip():
                return False
    
    # 3. Check that quality_score can be converted to float.
    try:
        score = float(example["quality_score"])
    except (TypeError, ValueError):
        return False
    if math.isnan(score) or math.isinf(score):
        return False
    
    # 4. Return True only if all checks pass.
    return True


def teacher_example_from_dict(example: dict) -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""
    # TODO:
    # 1. Validate the example.
    # 2. Raise ValueError if invalid.
    if not validate_teacher_example(example):
        raise ValueError(f"Invalid teacher example: {example!r}")

    # 3. Convert quality_score to float.
    # 4. Put any extra keys into metadata.
    metadata = {k: v for k, v in example.items() if k not in REQUIRED_FIELDS}

    # 5. Return a TeacherExample.
    return TeacherExample(
        prompt=example["prompt"],
        teacher_lyrics=example["teacher_lyrics"],
        teacher_notes=example["teacher_notes"],
        quality_score=float(example["quality_score"]),
        metadata=metadata
    )


def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    # TODO:
    # 1. Strip teacher_lyrics and teacher_notes.
    lyrics = teacher_lyrics.strip()
    notes = teacher_notes.strip()

    # 2. If include_thinking=True, return notes inside <think>...</think> tags,
    #    followed by a blank line and the lyrics.
    # 3. If include_thinking=False, return only the lyrics.
    # 4. Avoid extra leading/trailing whitespace.
    if include_thinking:
        return f"<think>{notes}</think>\n\n{lyrics}".strip()
    return lyrics


def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""
    # TODO:
    # 1. Strip the prompt.
    # 2. Return a consistent format such as:
    #    "User: Write a parody...\\nAssistant:"
    return f"User: {prompt.strip()}\nAssistant:"


def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""
    # TODO:
    # 1. Use format_training_prompt(prompt).
    # 2. Append target_text after the assistant marker.
    # 3. Return the full training string.
    return f"{format_training_prompt(prompt)} {target_text}".strip()


def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""
    # TODO:
    # 1. Call tokenizer.encode(text).
    # 2. Validate that the result is a list of integers.
    # 3. Return the token IDs.
    ids = tokenizer.encode(text)
    if not isinstance(ids, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in ids):
            raise ValueError("tokenizer.encode(text) must return a list of ints."
    )
    return ids


def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """
    # TODO:
    # 1. Copy input_ids into labels.
    # 2. If mask_prompt=True, replace labels before prompt_length with ignore_index.
    # 3. Return labels.
    labels = list(input_ids)
    if mask_prompt:
        n = max(0, min(prompt_length, len(labels)))
        labels[:n] = [ignore_index] * n
    return labels


def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""
    # TODO:
    # 1. Format the teacher response.
    target_text = format_teacher_response(
        example.teacher_lyrics,
        example.teacher_notes,
        include_thinking=include_thinking
    )

    # 2. Format the prompt.
    formatted_prompt = format_training_prompt(example.prompt)

    # 3. Build the full text.
    full_text = build_full_text(example.prompt, target_text)

    # 4. Tokenize the prompt and full text.
    prompt_ids = tokenize_text(tokenizer, formatted_prompt)
    input_ids = tokenize_text(tokenizer, full_text)
    prompt_length = len(prompt_ids)
    
    # 5. Create labels.
    labels = create_labels(input_ids, prompt_length, mask_prompt=mask_prompt)

    # 6. Return a DistillationRecord.
    metadata = dict(example.metadata)
    metadata["quality_score"] = example.quality_score

    return DistillationRecord(
        prompt=example.prompt.strip(),
        target_text=target_text,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=prompt_length,
        metadata=metadata
    )


def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4,
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""
    # TODO:
    kept = []

    # 1. Keep examples with quality_score >= min_quality_score.
    for ex in examples:
        if ex.quality_score < min_quality_score:
            continue

    # 2. Keep examples with at least min_lyric_lines non-empty lyric lines.
        text = ex.teacher_lyrics.replace("\\n", "\n")
        lyric_lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lyric_lines) < min_lyric_lines:
            continue
        kept.append(ex)

    # 3. Return the filtered list.
    return kept


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""
    # TODO:
    # 1. Validate validation_fraction is between 0 and 1.
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be between 0 and 1")

    # 2. Shuffle a copy of records using seed.
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    # 3. Split into train and validation records.
    val_count = int(len(shuffled) * validation_fraction)
    if len(shuffled) >= 2:
        val_count = min(max(val_count, 1), len(shuffled) -1)

    # 4. Return both lists.
    validation = shuffled[:val_count]
    train = shuffled[val_count:]
    return train, validation


def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""
    # TODO:
    # 1. Convert the dataclass to a dictionary.
    # 2. Return the dictionary.
    return asdict(record)


def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""
    # TODO:
    # 1. Open path for writing with UTF-8 encoding.
    # 2. Write one JSON object per line.
    # 3. Use record_to_json_dict for each record.
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record_to_json_dict(record), ensure_ascii=False))
            f.write("\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""
    # TODO:
    # 1. Open path with UTF-8 encoding.
    # 2. Read each non-empty line.
    # 3. Parse JSON and append to a list.
    # 4. Return the list.
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""
    # TODO:
    # 1. Handle empty records.
    if not records:
        return "Distillation records: 0 (no records to summarize)"

    # 2. Include number of records.
    # 3. Include average input length.
    # 4. Include average prompt length.
    # 5. Include average target length.
    n = len(records)
    avg_input = sum(len(r.input_ids) for r in records) / n
    avg_prompt = sum(r.prompt_length for r in records) / n
    avg_target = sum(len(r.input_ids) - r.prompt_length for r in records) /n

    return (
        f"Distillation records: {n}\n"
        f"Average input length (tokens): {avg_input:.1f}\n"
        f"Average prompt length (tokens): {avg_prompt:.1f}\n"
        f"Average target length (tokens): {avg_target:.1f}"
    )