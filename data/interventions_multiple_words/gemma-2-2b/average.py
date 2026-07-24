import json
from transformers import AutoTokenizer

langs = ["en", "de", "es", "fr", "zh", "ja", "ko"]
methods = ["description", "frequency", "value"]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    for lang1 in langs:
        for lang2 in langs:
            for method in methods:
                directory = f"{lang1}/{lang2}"
                base = f"interventions_and_results_{method}.json"
                destination = f"interventions_and_results_{method}_normalized.json"
                with open(f"{directory}/{base}", "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Normalize the data (example normalization, adjust as needed)
                for experiment, prompts in data.items():
                    for prompt, vals in prompts.items():
                        if "intervention" not in experiment:
                            for key1, val1 in vals.items():
                                for key2, val2 in val1.items():
                                    for options, val in val2.items():
                                        tokenized_prompt = tokenizer(options, return_tensors="pt")
                                        token_num = tokenized_prompt.input_ids.shape[1] - 1  # Exclude the special token
                                        val2[options] = val / token_num  # Normalize by the number of tokens
                        else:
                            for key1, val1 in vals.items():
                                for key2, val2 in val1.items():
                                    for key3, val3 in val2.items():
                                        for options, val in val3.items():
                                            tokenized_prompt = tokenizer(options, return_tensors="pt")
                                            token_num = tokenized_prompt.input_ids.shape[1] - 1  # Exclude the special token
                                            val3[options] = val / token_num  # Normalize by the number of tokens
                with open(f"{directory}/{destination}", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                print(f"Normalized {directory}/{base} and saved to {directory}/{destination}")
                            

