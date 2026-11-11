"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)

#helper method to save on formatting fluff
def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value)
    return str(value)

def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    # TODO:
    # 1. Read score_key from the evaluation dictionary.
    # 2. Convert it to a float.
    # 3. If the key is missing or cannot be converted, return 0.0.

    try:
        score = float(evaluation[score_key])
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0

    if math.isnan(score):
        return 0.0

    return score

def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    # TODO:
    # 1. Include the original prompt or topic.
    # 2. Include the current lyrics.
    # 3. Include the current overall score.
    # 4. Include any available evaluation details, such as rhyme, syllables, and structure.
    # 5. Include focus items, such as stronger rhymes or more consistent syllables.
    # 6. Include preserve items, such as topic, tone, and approximate length.
    # 7. Ask the model to return only the revised lyrics.
    #
    # Keep the prompt specific. "Make it better" is too vague.

    score = get_score(evaluation, score_key)
    focus_items = focus if focus else []
    preserve_items = preserve if preserve else []

    sections: list[str] = []

    sections.append("You are revising a parody song. Improve the lyrics below based on the evaluation feedback.")
    sections.append(f"Original prompt / topic:\n{original_prompt.strip()}")
    sections.append(f"Current lyrics:\n{current_text.strip()}")
    sections.append(f"Current {score_key}: {score:.3f}")

    detail_lines = [
        f"- {key}: {_format_value(value)}"
        for key, value in (evaluation or {}).items()
        if key != score_key and value is not None
    ]

    if detail_lines:
        sections.append("Evaluation details:\n" + "\n".join(detail_lines))
    if focus_items:
        sections.append(
            "Focus on these improvements:\n"
            + "\n".join(f"- {item}" for item in focus_items)
        )
    if preserve_items:
        sections.append(
            "Preserve these qualities:\n"
            + "\n".join(f"- {item}" for item in preserve_items)
        )
    sections.append(
        "Return only the revised lyrics. Do not include explanations, titles, notes, or commentary."
    )

    return "\n\n".join(sections) + "\n"
    

def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    # TODO:
    # 1. Evaluate current_text.
    current_eval = evaluation_func(current_text)
    prev_score = get_score(current_eval, score_key)

    # 2. Build a refinement prompt using build_refinement_prompt.
    prompt = build_refinement_prompt(
        original_prompt,
        current_text,
        current_eval,
        focus=focus,
        preserve=preserve,
        score_key=score_key
    )

    # 3. Generate revised text with generation_func.
    revised_text = generation_func(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )
    revised_text = revised_text.strip()

    # 4. Evaluate the revised text.
    revised_eval = evaluation_func(revised_text)
    revised_score = get_score(revised_eval, score_key)

    # 5. Compare the old score and revised score.
    # 6. Return a RefinementStep.
    #
    # This function should not decide whether to continue the full loop.
    # It only performs one attempt.

    return RefinementStep(
        iteration=0,
        prompt=prompt,
        previous_text=current_text,
        revised_text=revised_text,
        previous_score=prev_score,
        revised_score=revised_score,
        accepted=revised_score > prev_score,
        evaluation=revised_eval
    )


def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    # TODO:
    # 1. Compare the current score and revised score.
    # 2. If require_improvement is True, accept only strictly higher scores.
    # 3. If require_improvement is False, accept equal scores as well.
    # 4. Return the selected text, selected evaluation, and whether the revision was accepted.

    current_score = get_score(current_evaluation, score_key)
    revised_score = get_score(revised_evaluation, score_key)

    if require_improvement:
        accept = revised_score > current_score
    else:
        accept = revised_score >= current_score

    if accept:
        return revised_text, revised_evaluation, True
    return current_text, current_evaluation, False


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    # TODO:
    # 1. Stop if iteration has reached max_iterations.
    if iteration >= max_iterations:
        return False
    
    # 2. Stop if target_score is set and current_score is high enough.
    if target_score is not None and current_score >= target_score:
        return False
    
    # 3. Stop if stop_on_no_improvement is True and last_improved is False.
    if stop_on_no_improvement and not last_improved:
        return False
    
    # 4. Otherwise continue.
    return True


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    # TODO:
    # 1. Evaluate initial_text.
    starting_eval = evaluation_func(initial_text)
    starting_score = get_score(starting_eval, score_key)

    current_text = initial_text
    current_eval = starting_eval
    current_score = starting_score

    steps: list[RefinementStep] = []
    iteration = 0
    last_improved = True

    # 2. Repeatedly call refine_once until should_continue says to stop.
    while should_continue(
        iteration=iteration,
        max_iterations=max_iterations,
        current_score=current_score,
        target_score=target_score,
        last_improved=last_improved,
        stop_on_no_improvement=stop_on_no_improvement
    ):
        step = refine_once(
            original_prompt,
            current_text,
            generation_func,
            evaluation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            **kwargs
        )
        iteration += 1
        step.iteration = iteration

        # 3. After each step, keep the better text using choose_better_text.
        current_text, current_eval, step.accepted =  choose_better_text(
            current_text,
            step.revised_text,
            current_eval,
            step.evaluation,
            score_key=score_key,
            require_improvement=require_improvement
        )

        # 4. Track every RefinementStep in a list.
        steps.append(step)
        current_score = get_score(current_eval, score_key)
        last_improved = step.revised_score > step.previous_score

    # 5. Return a result dictionary with the initial and final information.
    #
    # Be careful to avoid infinite loops.
    if target_score is not None and current_score >= target_score:
        stopped_reason = "target_score"
    elif iteration >= max_iterations:
        stopped_reason = "max_iterations"
    else:
        stopped_reason = "no_improvement"

    return {
        "initial_text": initial_text,
        "initial_score": starting_score,
        "final_text": current_text,
        "final_score": current_score,
        "iterations": steps,
        "stopped_reason": stopped_reason
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    # TODO:
    # 1. Include the initial score and final score.
    # 2. Include the number of iterations.
    # 3. Include whether each iteration was accepted or rejected.
    # 4. Include the stopped reason.
    #
    # This is useful for debugging and for showing the tradeoff between extra
    # inference compute and output improvement.

    steps = result["iterations"]

    lines = [
        f"Initial score: {result['initial_score']:.3f}",
        f"Final score: {result['final_score']:.3f}",
        f"Iterations: {len(steps)}"
    ]

    for step in steps:
        status = "accepted" if step.accepted else "rejected"
        lines.append(f"    {step.iteration}: {step.previous_score:.3f} -> {step.revised_score:.3f} ({status})")

    lines.append(f"Stopped: {result['stopped_reason']}")

    return "\n".join(lines)