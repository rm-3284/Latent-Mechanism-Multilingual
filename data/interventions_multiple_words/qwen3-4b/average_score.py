import json
import numpy as np

if __name__ == "__main__":
    langs = ["en", "de", "es", "fr", "zh", "ja", "ko"]
    methods = ["description", "frequency", "value"]
    method = "description"

    diffs = {"en": [], "de": [], "es": [], "fr": [], "zh": [], "ja": [], "ko": []}
    for lang1 in langs:
        for lang2 in langs:
            if lang1 == lang2:
                continue
            else:
                directory = f"{lang1}/{lang2}"
                base = f"interventions_and_results_{method}_normalized.json"
                with open(f"{directory}/{base}", "r", encoding="utf-8") as f:
                    data = json.load(f)
                original = data["original"]
                intervention = data["feature-intervention"]
                diff = []
                for key, val in original.items():
                    before = val["logprobs"][lang2]
                    for before_key, before_val in before.items():
                        b = before_val
                    after = intervention[key][lang1][lang2][lang2]
                    for after_key, after_val in after.items():
                        a = after_val
                    diff.append(a - b)
                
                diffs[lang2].append(diff)
    for lang, diff in diffs.items():
        mean = np.mean(diff)
        print(f"{lang}: {mean}")
