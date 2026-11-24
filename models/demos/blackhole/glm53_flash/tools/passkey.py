"""Passkey retrieval prompts (the long-context and sequence-parallel tests, tools/sp_reference.py): filler sentences
with a passphrase sentence at a given depth, ending in a question whose answer is the passphrase."""

from __future__ import annotations

import torch

DEPTHS = [0.5, 0.1, 0.5, 0.8, 0.3, 0.6, 0.05, 0.5]
WORDS = ["amber", "falcon", "cobalt", "willow", "harbor", "saffron", "glacier", "meadow"]
TOPICS = [
    "the river",
    "the old library",
    "a new bridge",
    "the harvest",
    "the night market",
    "a mountain trail",
    "the city council",
    "a small bakery",
    "the train station",
    "the winter festival",
]


def filler(i: int) -> str:
    return (
        f"Note {i}: on day {i % 31 + 1} the people of town {i % 97} talked about {TOPICS[i % len(TOPICS)]}, "
        f"and the weather was {['mild', 'rainy', 'windy', 'clear', 'cold'][i % 5]}. "
    )


def prompt_ids(tok, length: int, depth: float, word: str, number: str) -> list[int]:
    """Filler sentences to about `length` tokens, the passphrase sentence at `depth` of them, then the question."""
    needle = f"The secret passphrase is {word}-{number}. Remember it. "
    question = (
        "\n\nQuestion: What is the secret passphrase mentioned in the notes above?\nAnswer: The secret passphrase is"
    )
    n_ids = lambda text: len(tok.encode(text).ids)  # noqa: E731
    per = n_ids(filler(0) + filler(1)) / 2
    budget = length - n_ids(needle + question) - 8
    count = max(1, int(budget / per))
    while True:
        sentences = [filler(i) for i in range(count)]
        sentences.insert(int(depth * count), needle)
        ids = tok.encode("".join(sentences) + question).ids
        if len(ids) <= length or count == 1:
            return ids
        count -= max(1, (len(ids) - length) // int(per) + 1)


# tests/test_model_device_prefill_sp.py's prompts: (length, depth, word, mesh row, user)
SP_CASES = [(600, 0.4, "amber", 2, 0), (1000, 0.2, "cobalt", 7, 1), (1900, 0.7, "harbor", 4, 1)]


def sp_prompts(tok) -> tuple[list[str], list[list[int]]]:
    """SP_CASES' answers and prompt ids (seeded: the same on the device and the CPU)."""
    g = torch.Generator().manual_seed(9)
    answers = [f"{w}-{int(torch.randint(1000, 9999, (1,), generator=g))}" for _, _, w, _, _ in SP_CASES]
    prompts = [prompt_ids(tok, n, d, *a.split("-")) for (n, d, *_), a in zip(SP_CASES, answers)]
    return answers, prompts
