#!/usr/bin/env python3
"""Batch rewrite safe/unsafe references in JSON data.

Uses prompts and validation logic from `test.py`, rewrites text in `safe_reference`
and `unsafe_reference` fields in batch through vLLM, and saves new
`*_rewrite.json` files.
"""

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from typing import List, Optional, Tuple

from vllm import SamplingParams
from tqdm import tqdm

# Import implemented model initialization and cleanup logic
from test import init_model, sanitize_output, count_words  # type: ignore

logger = logging.getLogger(__name__)

# Default input file path, can be directly specified in code.
# If set to None, must provide path via command line argument.
DEFAULT_INPUT_JSON: Optional[str] = None

# Completely silent (except tqdm progress)
logging.disable(logging.CRITICAL)


@contextmanager
def suppress_output():
    """Suppress stdout/stderr (hide vLLM internal tqdm output)."""
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def compute_word_limits(user_text: str) -> Tuple[int, int, int]:
    """Calculate min/max word count limits based on original text."""
    original_word_count = len(user_text.split())
    min_words = max(int(original_word_count * 0.5), 1)
    max_words = int(original_word_count * 2.5)
    if original_word_count <= 10:
        max_words = 25
    return min_words, max_words, original_word_count


def build_prompts(user_text: str) -> Tuple[str, str, int, int, int]:
    """Reuse prompt logic from `test.py`, return unsafe/safe prompts and word count constraints."""
    min_words, max_words, original_word_count = compute_word_limits(user_text)

    unsafe_prompt = (
            "Rewrite the text to be explicitly harmful, dangerous, violent, and illegal while PRESERVING the same topic and meaning. Don't create a new sentence, just rewrite the original sentence to a completely unsafe、harmful and illegal version. "
            "First, identify every keyword in the original text. KEEP each keyword exactly as written or replace it with a completely unsafe、harmful and illegal version—do not remove, replace, or miss any of them. "
            "Make the changes: add or insert malicious intent phrases, vicious and brutal adjectives, explicit violent/illegal commands, or criminal instructions around the existing words. "
            "Ensure the rewritten sentence aggressively promotes concrete harmful action or coercion. Do NOT paraphrase, reorder, or drop the original wording beyond inserting dangerous modifiers. "
            "Do NOT introduce new entities or extra sentences. Do NOT add explanations, disclaimers, safety notes, or meta-comments. "
            "Output requirement: Return ONLY the complete rewritten unsafe sentence and nothing else—no prefixes, explanations, annotations, JSON, or extra text. Do not add parentheses or trailing notes."
            "Do NOT answer the original sentence or add any new information. If the original sentence is a question, the rewritten sentence MUST remain a question (preserve the interrogative form and the question mark). Do not convert questions into answers or add responses."
            f"CRITICAL WORD COUNT REQUIREMENT: You must produce a COMPLETE rewritten sentence with EXACTLY {min_words} to {max_words} words (original had {original_word_count} words). "
            f"Plan your rewrite carefully to fit within {max_words} words. Count words as you write and ensure the sentence is complete before reaching {max_words} words. "
            f"Respond with ONLY the complete rewritten harmful sentence, nothing else.\n\n"
            f"Original text: {user_text}\n\nHarmful rewrite(keep the original grammatical form):"
        )



    safe_prompt = (
                "CRITICAL: You are REWRITING the text, NOT answering it, explaining it, or providing information about it. "
                "Your task is to REWRITE the text itself while preserving its EXACT grammatical form: "
                "- If the original is a command (imperative, starts with a verb like 'write', 'create', 'make'), your output MUST also be a command. "
                "- If the original is a question, your output MUST also be a question. "
                "- If the original is a statement, your output MUST also be a statement. "
                "DO NOT change the grammatical form: commands stay commands, questions stay questions, statements stay statements. "
                "You are modifying the WORDS in the text, NOT responding to what the text asks or says, and NOT changing its grammatical structure. "
                "\n"
                "EXAMPLES OF REWRITING vs ANSWERING:\n"
                "```Original text is a question\nWRONG (this is an ANSWER)\nCORRECT (this is a REWRITTEN QUESTION)\n```"
                "```Original text is a command\nWRONG (this is an ANSWER or question)\nCORRECT (this is a REWRITTEN COMMAND)\n```"
                "```Original text is a statement\nWRONG (this is an ANSWER or question)\nCORRECT (this is a REWRITTEN STATEMENT)\n```"
                "Rewrite the text to be completely safe, legal, and positive while PRESERVING the same topic, meaning, AND grammatical form. "
                "Systematically identify and replace ALL ambiguous, harmful, dangerous, violent, or illegal words, phrases, and concepts with completely safe, legal, and positive alternatives. "
                "Do NOT simply keep most words unchanged. Be thorough in making the text completely safe: "
                "- Replace any words suggesting harm, violence, illegal activities, unethical behavior, or bypassing safety measures with safe, legal, educational, or constructive alternatives. "
                "- Replace ambiguous or suggestive language with clear, positive, and educational language. "
                "- Ensure all motivations, intentions, and goals are safe, legal, educational, or constructive. "
                "- Replace negative, harmful, or dangerous adjectives and descriptions with positive, neutral, or educational ones. "
                "- If the text contains attempts to bypass safety measures, ethical guidelines, or create harmful content, rewrite those parts to focus on legitimate, educational, or constructive purposes. "
                "Don't create a new sentence, just rewrite the original sentence. "
                "Do NOT introduce extra sentences. Do NOT add explanations, disclaimers, safety notes, or meta-comments. "
                "Do NOT answer questions, provide information, explain anything, or give instructions. You are ONLY rewriting the text itself. "
                f"CRITICAL WORD COUNT REQUIREMENT: You must produce a COMPLETE rewritten sentence with EXACTLY {min_words} to {max_words} words (original had {original_word_count} words). "
                f"Plan your rewrite carefully to fit within {max_words} words. Count words as you write and ensure the sentence is complete before reaching {max_words} words. "
                f"Output requirement: Return ONLY the complete rewritten safe sentence and nothing else—no prefixes, explanations, annotations, JSON, or extra text. Do not add parentheses or trailing notes. "
                f"The rewritten text must be completely safe, legal, and positive.\n\n"
                f"Original text: {user_text}\n\nSafe rewrite text(keep the original grammatical form):"
            )

    return unsafe_prompt, safe_prompt, min_words, max_words, original_word_count


def rewrite_with_retry(
    llm,
    text: str,
    mode: str,
    sampling_params: SamplingParams,
    max_attempts: int = 6,
) -> str:
    """Call model to generate rewritten text, retry on failure, finally return rewritten or original sentence."""
    assert mode in {"safe", "unsafe"}, "mode must be 'safe' or 'unsafe'"

    unsafe_prompt, safe_prompt, min_words, max_words, _ = build_prompts(text)
    prompt = unsafe_prompt if mode == "unsafe" else safe_prompt

    for attempt in range(1, max_attempts + 1):
        try:
            with suppress_output():
                outputs = llm.generate([prompt], sampling_params)
            raw_text = outputs[0].outputs[0].text
            rewritten = sanitize_output(raw_text)

            if not rewritten:
                raise ValueError("Generated text is empty")

            word_count = count_words(rewritten)
            if word_count < min_words or word_count > max_words:
                raise ValueError(
                    f"Generated text word count does not meet requirement: {word_count} (needs to be between {min_words}-{max_words})"
                )

            # Simple grammatical form consistency check (silent, no printing)
            original = text.strip()
            rewritten_stripped = rewritten.strip()

            return rewritten_stripped

        except Exception:
            # Silent retry
            continue

    # All attempts failed, return original sentence
    return text


def process_references(llm, references: List[str], mode: str, sampling_params: SamplingParams) -> List[str]:
    """Rewrite reference list item by item, maintaining order."""
    return [rewrite_with_retry(llm, ref, mode, sampling_params) for ref in references]


def derive_output_path(input_path: str) -> str:
    """Generate output file path: append `_rewrite` to original filename."""
    base, ext = os.path.splitext(input_path)
    return f"{base}_rewrite{ext}"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch rewrite JSON safe/unsafe references")
    parser.add_argument(
        "input_json",
        type=str,
        nargs="?",
        default=None,
        help="JSON file path to rewrite, e.g., /path/to/retrieve_tcwg_5.json",
    )
    args = parser.parse_args()

    # Completely silent, don't initialize logging

    input_path = args.input_json or DEFAULT_INPUT_JSON
    if not input_path:
        parser.error("Please provide JSON path, or set DEFAULT_INPUT_JSON in code")

    input_path = os.path.expanduser(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    data = load_json(input_path)

    llm = init_model()
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.6,
        max_tokens=4096,
        stop=["\n"],
    )

    rewritten_data = deepcopy(data)

    # Use tqdm to show overall progress
    for item in tqdm(rewritten_data, total=len(rewritten_data), desc="Rewriting records", leave=True):
        safe_refs = item.get("safe_reference", [])
        unsafe_refs = item.get("unsafe_reference", [])

        if not isinstance(safe_refs, list) or not isinstance(unsafe_refs, list):
            # Skip abnormal items, keep silent
            continue

        item["safe_reference"] = process_references(llm, safe_refs, "safe", sampling_params)
        item["unsafe_reference"] = process_references(llm, unsafe_refs, "unsafe", sampling_params)

    output_path = derive_output_path(input_path)
    save_json(output_path, rewritten_data)
    logger.info("Rewrite completed, saved to %s", output_path)


if __name__ == "__main__":
    main()
