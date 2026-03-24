"""
perturbations.py

Phase 3 — Input perturbation pipeline.

Implements 5 deterministic perturbations that simulate real-world noise
encountered in production. All perturbations modify only the user message
content; message structure and reference answer are unchanged so the existing
evaluation pipeline can score the perturbed outputs against the same reference.

Perturbation types
------------------
  typo         Character swaps, deletions, and keyboard-neighbor substitutions.
  noise        Prepend/append irrelevant context sentences.
  truncation   Drop the last fraction of words (simulates incomplete input).
  case_lower   Convert entire user message to lowercase.
  paraphrase   Shuffle sentence order (deterministic, no LLM required).

Public API
----------
  perturb_sample(sample, perturbation_type, seed) -> dict
  perturb_dataset(samples, perturbation_type, seed) -> list[dict]
  PERTURBATION_TYPES -> list[str]
"""

from __future__ import annotations

import copy
import random
import re

# ---------------------------------------------------------------------------
# QWERTY keyboard neighbor map  (lowercase only)
# ---------------------------------------------------------------------------

_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "sqwz",  "b": "vghn",  "c": "xdfv",  "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko",  "j": "huikmn",
    "k": "jiolm",  "l": "kop",   "m": "njk",   "n": "bhjm",   "o": "iklp",
    "p": "ol",     "q": "wa",    "r": "edft",  "s": "awedxz", "t": "rfgy",
    "u": "yhji",   "v": "cfgb",  "w": "qase",  "x": "zsdc",   "y": "tghu",
    "z": "asx",
}

# Noise sentences that could plausibly appear in a real support ticket
_NOISE_SENTENCES = [
    "Please process this as soon as possible.",
    "This ticket was escalated from the on-call team.",
    "The system has been experiencing intermittent issues.",
    "All previous attempts to resolve this have failed.",
    "Refer to the incident tracking system for more context.",
    "This issue was first reported by the monitoring dashboard.",
    "High-priority incident — SLA breach imminent.",
    "Standard operating procedures have already been followed.",
    "This was flagged during the weekly reliability review.",
    "The engineering lead has been notified.",
]


# ---------------------------------------------------------------------------
# Individual perturbation functions
# ---------------------------------------------------------------------------

def apply_typo(text: str, rate: float = 0.05, seed: int = 42) -> str:
    """
    Introduce character-level noise at the given rate.

    For each alphabetic character, with probability `rate`, one of three
    operations is applied: swap with the next character, delete it, or
    replace it with a keyboard-adjacent character.

    Args:
        text: Input string.
        rate: Per-character error probability (default 5%).
        seed: Random seed for reproducibility.

    Returns:
        Text with character-level noise introduced.
    """
    if not text:
        return text

    rng = random.Random(seed)
    chars = list(text)
    result: list[str] = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isalpha() and rng.random() < rate:
            action = rng.choice(["swap", "delete", "substitute"])
            if action == "swap" and i + 1 < len(chars):
                result.append(chars[i + 1])
                result.append(ch)
                i += 2
                continue
            elif action == "delete":
                i += 1
                continue
            elif action == "substitute":
                neighbors = _KEYBOARD_NEIGHBORS.get(ch.lower(), "")
                if neighbors:
                    replacement = rng.choice(neighbors)
                    result.append(replacement if ch.islower() else replacement.upper())
                    i += 1
                    continue
        result.append(ch)
        i += 1
    return "".join(result)


def apply_noise(text: str, position: str = "prefix", seed: int = 42) -> str:
    """
    Add an irrelevant context sentence before or after the actual content.

    Simulates tickets where support staff prepend/append boilerplate text
    that is unrelated to the actual technical issue.

    Args:
        text:     Input string.
        position: Where to inject — "prefix", "suffix", or "both".
        seed:     Random seed.

    Returns:
        Text with injected noise sentence(s).
    """
    rng = random.Random(seed)
    sentence = rng.choice(_NOISE_SENTENCES)
    if position == "suffix":
        return f"{text} {sentence}"
    elif position == "both":
        remaining = [s for s in _NOISE_SENTENCES if s != sentence]
        sentence2 = rng.choice(remaining or _NOISE_SENTENCES)
        return f"{sentence} {text} {sentence2}"
    else:  # default: prefix
        return f"{sentence} {text}"


def apply_truncation(text: str, drop_frac: float = 0.3, seed: int = 42) -> str:
    """
    Drop the last `drop_frac` fraction of words from the text.

    Simulates incomplete inputs where the user stops typing mid-sentence
    or the front-end cuts off the message.

    Args:
        text:      Input string.
        drop_frac: Fraction of words to remove from the end (default 30%).
        seed:      Unused — truncation is deterministic, kept for API consistency.

    Returns:
        Truncated text (always retains at least 3 words).
    """
    words = text.split()
    if len(words) <= 3:
        return text
    keep = max(3, int(len(words) * (1.0 - drop_frac)))
    return " ".join(words[:keep])


def apply_case_lower(text: str) -> str:
    """
    Convert the entire message to lowercase.

    Simulates users who disable autocapitalization or paste from terminals.

    Args:
        text: Input string.

    Returns:
        Fully lowercased text.
    """
    return text.lower()


def apply_paraphrase(text: str, seed: int = 42) -> str:
    """
    Shuffle sentence order within the text.

    A lightweight deterministic paraphrase that tests whether the model
    relies on sentence order rather than overall semantics. No LLM required.

    Args:
        text: Input string.
        seed: Random seed for the shuffle.

    Returns:
        Text with sentences in shuffled order. Single-sentence inputs are
        returned unchanged.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text
    rng = random.Random(seed)
    rng.shuffle(sentences)
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_PERTURBATION_FNS: dict = {
    "typo":       lambda text, seed: apply_typo(text, rate=0.05, seed=seed),
    "noise":      lambda text, seed: apply_noise(text, position="prefix", seed=seed),
    "truncation": lambda text, seed: apply_truncation(text, drop_frac=0.30, seed=seed),
    "case_lower": lambda text, seed: apply_case_lower(text),
    "paraphrase": lambda text, seed: apply_paraphrase(text, seed=seed),
}

PERTURBATION_TYPES: list[str] = list(_PERTURBATION_FNS.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perturb_sample(sample: dict, perturbation_type: str, seed: int = 42) -> dict:
    """
    Apply a perturbation to all user messages in a sample.

    Creates a deep copy — the original sample is never modified.
    Records the perturbation type in sample metadata for traceability.

    Args:
        sample:            Sample dict with at least a "messages" key.
        perturbation_type: One of PERTURBATION_TYPES.
        seed:              Random seed for reproducibility.

    Returns:
        New sample dict with perturbed user messages.

    Raises:
        ValueError: If perturbation_type is not in PERTURBATION_TYPES.
    """
    if perturbation_type not in _PERTURBATION_FNS:
        raise ValueError(
            f"Unknown perturbation: '{perturbation_type}'. "
            f"Valid types: {PERTURBATION_TYPES}"
        )

    perturb_fn = _PERTURBATION_FNS[perturbation_type]
    perturbed = copy.deepcopy(sample)

    for msg in perturbed.get("messages", []):
        if msg.get("role") == "user":
            original = msg.get("content", "")
            msg["content"] = perturb_fn(original, seed)

    # Tag metadata for traceability
    if not isinstance(perturbed.get("metadata"), dict):
        perturbed["metadata"] = {}
    perturbed["metadata"]["perturbation"] = perturbation_type
    perturbed["metadata"]["perturbation_seed"] = seed

    return perturbed


def perturb_dataset(
    samples: list[dict],
    perturbation_type: str,
    seed: int = 42,
) -> list[dict]:
    """
    Apply a perturbation to every sample in a dataset.

    The seed is incremented per sample to introduce variety while keeping
    the full run reproducible.

    Args:
        samples:           List of sample dicts.
        perturbation_type: One of PERTURBATION_TYPES.
        seed:              Base random seed (incremented per sample).

    Returns:
        List of perturbed sample dicts (same length as input).
    """
    return [
        perturb_sample(sample, perturbation_type, seed=seed + i)
        for i, sample in enumerate(samples)
    ]
