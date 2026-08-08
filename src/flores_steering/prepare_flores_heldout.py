"""
Prepare held-out FLORES+ sentences for cross-lingual continuation steering.

Selects sentences from indices not used during latent extraction (default: 150-154
after shuffle seed=42; extraction uses indices 0-149). Loads parallel translations
for all languages in lang_to_flores_key and truncates each sentence at the token
midpoint.

Example:
  python prepare_flores_heldout.py --model gemma-2-2b
  python prepare_flores_heldout.py --model gemma-2-2b --heldout-start 150 --num-sentences 5
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from sensitivity.hyperparameter_sensitivity import repo_data_dir
from lib.models import hf_model_names, use_bos
from lib.pipeline_data.generic_sentences import alphabet_char
from lib.template import lang_to_flores_key


def truncate_at_midpoint(
    sentence: str,
    tokenizer,
    lang: str,
    model_name: str,
    min_tokens: int = 6,
) -> tuple[str, str, int]:
    """Return (prefix, suffix, cut_token_index) for a midpoint token split."""
    tokenized = tokenizer.encode(sentence)
    n = len(tokenized)
    if n <= min_tokens:
        raise ValueError(f"Sentence too short ({n} tokens): {sentence!r}")

    char_fn = alphabet_char[lang]
    cut = n // 2
    iterations = 0
    while cut < n - 2 and not _cut_is_valid(tokenized, cut, tokenizer, char_fn):
        cut += 1
        iterations += 1
        if iterations > n:
            raise ValueError(f"Could not find valid midpoint cut for: {sentence!r}")

    skip_bos = 1 if use_bos.get(model_name, True) else 0
    prefix = tokenizer.decode(tokenized[skip_bos:cut])
    suffix = tokenizer.decode(tokenized[cut:])
    return prefix, suffix, cut


def _cut_is_valid(tokenized: list[int], cut: int, tokenizer, char_fn) -> bool:
    next_token = tokenizer.decode(tokenized[cut : cut + 1])
    word = next_token
    if word and word[0] == " " and len(word) > 1:
        return char_fn(word[1])
    if word:
        return char_fn(word[0])
    return False


def load_parallel_sentences(
    indices: list[int],
    shuffle_seed: int,
) -> dict[int, dict[str, str]]:
    """Load full FLORES+ dev sentences at given shuffled indices for all langs."""
    by_index: dict[int, dict[str, str]] = {idx: {} for idx in indices}
    for lang, ds_key in lang_to_flores_key.items():
        ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
        ds = ds.shuffle(seed=shuffle_seed)
        for idx in indices:
            by_index[idx][lang] = ds[idx]["text"]
    return by_index


def build_heldout_payload(
    model_name: str,
    heldout_start: int,
    num_sentences: int,
    extraction_end: int,
    shuffle_seed: int,
) -> dict:
    indices = list(range(heldout_start, heldout_start + num_sentences))
    if indices[0] < extraction_end:
        raise ValueError(
            f"Held-out indices {indices[0]}-{indices[-1]} overlap extraction pool "
            f"(0-{extraction_end - 1}). Increase --heldout-start."
        )

    hf_name = hf_model_names[model_name]
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    parallel = load_parallel_sentences(indices, shuffle_seed)

    sentences = []
    for idx in indices:
        entry: dict = {"flores_index": idx, "languages": {}}
        for lang, full_text in parallel[idx].items():
            prefix, suffix, cut = truncate_at_midpoint(
                full_text, tokenizer, lang, model_name
            )
            entry["languages"][lang] = {
                "full": full_text,
                "prefix": prefix,
                "suffix": suffix,
                "cut_token": cut,
            }
        sentences.append(entry)

    return {
        "model": model_name,
        "shuffle_seed": shuffle_seed,
        "extraction_indices": f"0-{extraction_end - 1}",
        "heldout_indices": indices,
        "languages": list(lang_to_flores_key.keys()),
        "sentences": sentences,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare held-out FLORES+ continuation data")
    parser.add_argument("--model", type=str, default="gemma-2-2b", choices=hf_model_names.keys())
    parser.add_argument("--heldout-start", type=int, default=150)
    parser.add_argument("--num-sentences", type=int, default=5)
    parser.add_argument("--extraction-end", type=int, default=150)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_heldout_payload(
        model_name=args.model,
        heldout_start=args.heldout_start,
        num_sentences=args.num_sentences,
        extraction_end=args.extraction_end,
        shuffle_seed=args.shuffle_seed,
    )
    out_dir = os.path.join(repo_data_dir(), "flores_heldout", args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, "heldout_sentences.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(payload['sentences'])} held-out sentences to {out_path}")


if __name__ == "__main__":
    main()
