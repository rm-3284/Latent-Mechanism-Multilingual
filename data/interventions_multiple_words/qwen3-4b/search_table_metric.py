#!/usr/bin/env python3
"""Search JSON key combinations that reproduce the paper Enumerations table."""
import json
import math
import os
from itertools import product

import numpy as np

LANGS = ["en", "de", "es", "fr", "zh", "ja", "ko"]
TABLE = {
    "value": [0.89, -0.73, 0.25, 0.95, 0.40, 0.97, 0.61, 0.48],
    "frequency": [1.50, 0.28, 1.41, 1.90, 1.26, 1.49, 1.81, 1.38],
    "description": [0.73, None, 0.17, 0.82, 0.18, 0.86, 0.49, 0.46],
}
ROOT = os.path.dirname(os.path.abspath(__file__))


def scalar(d):
    return next(iter(d.values())) if isinstance(d, dict) else d


def score(method, out):
    targets = TABLE[method]
    err = 0.0
    for i, lang in enumerate(LANGS):
        t = targets[i]
        if t is None or lang not in out or math.isnan(out[lang]):
            continue
        err += (out[lang] - t) ** 2
    mean_target = targets[7]
    lang_vals = [out[l] for l in LANGS if l in out and not math.isnan(out[l])]
    if mean_target is not None and lang_vals:
        err += (float(np.mean(lang_vals)) - mean_target) ** 2
    return err


def load_all_pairs(method, norm):
    suf = "_normalized" if norm else ""
    cache = {}
    for lang1 in LANGS:
        for lang2 in LANGS:
            if lang1 == lang2:
                continue
            path = os.path.join(
                ROOT,
                f"{lang1}/{lang2}/interventions_and_results_{method}{suf}.json",
            )
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                cache[(lang1, lang2)] = json.load(f)
    return cache


def accumulate_fi(cache, abl, amp, meas):
    vals = {t: [] for t in LANGS}
    for (lang1, lang2), data in cache.items():
        exp = data.get("feature-intervention")
        if not exp:
            continue
        for prompt, oval in data["original"].items():
            if prompt not in exp:
                continue
            try:
                after = scalar(exp[prompt][abl][amp][meas])
                before = scalar(oval["logprobs"][meas])
            except (KeyError, TypeError, StopIteration):
                continue
            if math.isfinite(after) and math.isfinite(before):
                vals[meas].append(after - before)
    return vals


def accumulate_other(cache, exp, key_a, key_b, meas):
    vals = {t: [] for t in LANGS}
    for (_lang1, _lang2), data in cache.items():
        block_exp = data.get(exp)
        if not block_exp:
            continue
        for prompt, oval in block_exp.items():
            if prompt == "original" or prompt not in data["original"]:
                continue
        for (_lang1, _lang2), data in cache.items():
            block_exp = data.get(exp)
            if not block_exp:
                continue
            for prompt, oval in data["original"].items():
                if prompt not in block_exp:
                    continue
                try:
                    if exp == "distractor ablation":
                        after = scalar(block_exp[prompt][key_a][key_b])
                    else:
                        after = scalar(block_exp[prompt][key_a][key_b])
                    before = scalar(oval["logprobs"][meas])
                except (KeyError, TypeError, StopIteration):
                    continue
                if math.isfinite(after) and math.isfinite(before):
                    vals[meas].append(after - before)
    return vals


def main():
    best = []
    for method in TABLE:
        for norm in (False, True):
            cache = load_all_pairs(method, norm)
            if not cache:
                continue
            for abl, amp, meas in product(LANGS, repeat=3):
                vals = accumulate_fi(cache, abl, amp, meas)
                if not any(vals[t] for t in LANGS):
                    continue
                out = {
                    t: float(np.mean(vals[t])) if vals[t] else float("nan")
                    for t in LANGS
                }
                err = score(method, out)
                if err < 5.0:
                    gmean = round(
                        float(np.mean([out[t] for t in LANGS if not math.isnan(out[t])])),
                        2,
                    )
                    best.append(
                        (
                            err,
                            method,
                            norm,
                            abl,
                            amp,
                            meas,
                            {t: round(out[t], 2) for t in LANGS},
                            gmean,
                        )
                    )

    best.sort(key=lambda x: x[0])
    print(f"feature-intervention triples with err < 5: {len(best)}\n")
    for row in best[:25]:
        err, method, norm, abl, amp, meas, out, gmean = row
        print(
            f"err={err:.2f} {method} norm={norm} ({abl},{amp},{meas}) -> {out} gmean={gmean}"
        )
        print(f"  target langs+mean: {TABLE[method]}\n")


if __name__ == "__main__":
    main()
