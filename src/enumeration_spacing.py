"""Locale-aware separators for enumeration prompts and multiple_words.json samples."""

from __future__ import annotations

# Order used in multiple_words.json "samples" blocks (must match file "languages" subset).
SAMPLE_LANG_ORDER = ("en", "fr", "de", "es", "zh", "ja", "ko", "ar", "ru", "hi", "pt", "it", "nl", "pl")


def enumeration_join(items: list[str], lang: str) -> str:
    if not items:
        return ""
    if lang == "zh":
        return "，".join(items)
    if lang == "ja":
        return "、".join(items)
    if lang == "ar":
        return "، ".join(items)
    return ", ".join(items)


def enumeration_suffix(lang: str) -> str:
    """Trailing punctuation after the last displayed item (before model continuation)."""
    if lang == "zh":
        return "，"
    if lang == "ja":
        return "、"
    if lang == "ar":
        return "،"
    return ","


def prompt_fill_in(display_choices: list[str], list_lang: str) -> str:
    return enumeration_join(display_choices, list_lang) + enumeration_suffix(list_lang)


def continuation_target(rest: list[str], tgt_lang: str) -> str:
    """String matched against model continuation (matches JSON sample targets)."""
    body = enumeration_join(rest, tgt_lang)
    if tgt_lang in ("zh", "ja"):
        return body
    return " " + body


def build_all_samples(translations: dict[str, dict], split_at: int) -> dict:
    samples: dict[str, dict] = {}
    for src in SAMPLE_LANG_ORDER:
        frame = translations[src]["frame"]
        seq_src = translations[src]["sequence"]
        ctx = frame + enumeration_join(seq_src[:split_at], src) + enumeration_suffix(src)
        targets = {
            tgt: continuation_target(translations[tgt]["sequence"][split_at:], tgt)
            for tgt in SAMPLE_LANG_ORDER
            if tgt != src
        }
        samples[src] = {"context": ctx, "targets": targets}
    return samples
