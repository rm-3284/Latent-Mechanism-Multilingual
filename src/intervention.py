import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from circuit_tracer_import import Feature, ReplacementModel


def _first_content_token_id(model: ReplacementModel, text: str) -> int:
    token_ids = model.tokenizer.encode(text)
    if not token_ids:
        raise ValueError(f"Tokenizer produced empty tokenization for text: {text}")

    special_ids = set(getattr(model.tokenizer, "all_special_ids", []))
    for token_id in token_ids:
        if token_id not in special_ids:
            return token_id

    # Fallback when tokenizer only returns special tokens with defaults.
    plain_ids = model.tokenizer.encode(text, add_special_tokens=False)
    if not plain_ids:
        raise ValueError(f"Tokenizer produced only special tokens for text: {text}")
    return plain_ids[0]

def ablation(supernode_dict: dict[str, list[tuple[Feature, float]]], lang: str, alpha: float=0) -> list[tuple[int, int, int, torch.Tensor]]:
    if lang not in supernode_dict.keys():
        raise KeyError(f'{lang} is not a valid intervention language.')
    intervention_values = []
    for feature, val in supernode_dict[lang]:
        if val == float("nan"):
            continue
        layer = feature.layer
        pos = feature.pos
        feature_idx = feature.feature_idx

        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        pos = pos.item() if isinstance(pos, torch.Tensor) else pos
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx

        ablation_value = torch.tensor(val * alpha)
        intervention_values.append((layer, pos, feature_idx, ablation_value))
    return intervention_values

def amplification(supernode_dict: dict[str, list[tuple[Feature, float]]], lang: str) -> list[tuple[int, int, int, torch.Tensor]]:
    if lang not in supernode_dict.keys():
        raise KeyError(f'{lang} is not a valid intervention language.')
    intervention_values = []
    for feature, val in supernode_dict[lang]:
        if val == float("nan"):
            continue
        layer = feature.layer
        pos = feature.pos
        feature_idx = feature.feature_idx

        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        pos = pos.item() if isinstance(pos, torch.Tensor) else pos
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx

        amplification_value = torch.tensor(val) if isinstance(val, float) else val
        intervention_values.append((layer, pos, feature_idx, amplification_value))
    return intervention_values


def get_best_base(logits: torch.Tensor, targets: list[str], model: ReplacementModel) -> str:
    last_logits = logits.squeeze(0)[-1]
    _, indices = torch.sort(last_logits, dim=-1, descending=True)
    ranks = []
    for target in targets:
        token = _first_content_token_id(model, target)
        mask = (indices == token)
        rank = torch.argmax(mask.int(), dim=-1)
        rank = rank.item() if isinstance(rank, torch.Tensor) else rank
        ranks.append((target, rank))
    ranks.sort(key=lambda x: x[1], reverse=False)
    return ranks[0][0]

def get_best_rank(logits: torch.Tensor, targets: list[str], model: ReplacementModel) -> int:
    last_logits = logits.squeeze(0)[-1]
    _, indices = torch.sort(last_logits, dim=-1, descending=True)
    ranks = []
    for target in targets:
        token = _first_content_token_id(model, target)
        mask = (indices == token)
        rank = torch.argmax(mask.int(), dim=-1)
        rank = rank.item() if isinstance(rank, torch.Tensor) else rank
        ranks.append(rank)
    return min(ranks)

def get_top_outputs(logits: torch.Tensor, model: ReplacementModel, k: int = 10):
    top_probs, top_token_ids = logits.squeeze(0)[-1].softmax(-1).topk(k)
    top_tokens = [model.tokenizer.decode(token_id) for token_id in top_token_ids]
    top_outputs = list(zip(top_tokens, top_probs.tolist()))
    return top_outputs

def logit_diff_single(old_logits: torch.Tensor, new_logits: torch.Tensor, target: str, base: str, model: ReplacementModel) -> tuple[float, float, float]:
    o_logits = old_logits.squeeze(0)[-1]
    n_logits = new_logits.squeeze(0)[-1]

    s = _first_content_token_id(model, base)
    t = _first_content_token_id(model, target)

    o_diff = o_logits[t] - o_logits[s]
    n_diff = n_logits[t] - n_logits[s]
    o_diff = o_diff.item() if isinstance(o_diff, torch.Tensor) else o_diff
    n_diff = n_diff.item() if isinstance(n_diff, torch.Tensor) else n_diff

    diff = n_diff - o_diff

    print(f'Logit difference of "{target}" to "{base}": old {o_diff}, new {n_diff}, diff {diff}')
    return o_diff, n_diff, diff

def check_valid_meaning(prompt: str, ans: dict[str, list[str]], model: ReplacementModel, k: int=10) -> bool:
    logits, activations = model.get_activations(prompt)
    en = get_best_rank(logits, ans['en'], model)
    zh = get_best_rank(logits, ans['zh'], model)
    return en <= k or zh <= k

def visualize_bar_2ddict_outer_inter(data: dict[str, dict[str, float]], interactive=True, plt_path=''):
    # I will name the keys outer and inter
    
    outer_keys = list(data.keys())
    # Sort them for consistent order
    outer_keys_sorted = sorted(outer_keys)
    num_outer_keys = len(outer_keys_sorted)

    # Get all unique inner keys for the grouped bars and legend
    inter_keys = set()
    for d in data.values():
        inter_keys.update(d.keys())
    # Sort them to ensure consistent order in the legend and bar groups
    inter_keys_sorted = sorted(list(inter_keys))
    num_inter_keys = len(inter_keys_sorted)

    plt.figure(figsize=(12, 7)) # Set a larger figure size for better readability

    total_group_width = 0.8
    bar_width = total_group_width / num_inter_keys # Width of each individual bar within a group
    # Base positions for each group of bars on the x-axis
    index = np.arange(num_outer_keys)

    # Define a color palette for each output language
    colors = cm.get_cmap('tab10', num_inter_keys)

    # Iterate through each output language (lang2) to plot its bars across prompt languages
    for i, inter_key in enumerate(inter_keys_sorted):
        # Calculate the offset for the current output language's bars
        # This centers the group of bars around the tick mark for each prompt language
        offset = (i - (num_inter_keys - 1) / 2) * bar_width

        # Get the likelihood scores for the current output_lang across all prompt languages
        values = [data[outer_key].get(inter_key, 0) for outer_key in outer_keys_sorted]

        # Plot the bars
        plt.bar(index + offset, values, bar_width, label=inter_key, color=colors(i % num_inter_keys))

    for i in range(num_outer_keys - 1):
        # The x-position for the vertical line is half-way between the current intervention type's
        # x-index and the next one's x-index.
        # Since 'index' is `np.arange(len(intervention_types_sorted))`, the points are 0, 1, 2...
        # A line between index[i] and index[i+1] is at (index[i] + index[i+1]) / 2, which simplifies to i + 0.5.
        plt.axvline(x=i + 0.5, color='gray', linestyle='-', linewidth=1)

    plt.xlabel('Prompt Language (lang1)', fontsize=12)
    plt.ylabel('Likelihood Score', fontsize=12)
    plt.title('LLM Output Likelihood by Prompt Language and Output Language', fontsize=14)

    # Set x-axis ticks to be at the center of each group of bars
    plt.xticks(index, outer_keys_sorted, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend(title='Output Language (lang2)', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    if interactive:
        plt.show()
    else:
        plt.savefig(plt_path)
        plt.close()


def bar_graph_visualize(data: dict[str, dict[str, dict[str, float]]], name:str, interactive=True, plt_path='') -> None:
    # Get the outermost keys (lang1: e.g., 'English Prompt', 'Spanish Prompt')
    prompt_languages = sorted(list(data.keys()))

    # Get the innermost keys (lang2: e.g., 'French Output', 'German Output') for legend and individual bars
    # Assuming all innermost dictionaries have the same keys for consistency in plotting
    first_prompt_lang = prompt_languages[0]
    first_intervention_type = list(data[first_prompt_lang].keys())[0]
    output_languages = sorted(list(data[first_prompt_lang][first_intervention_type].keys()))
    num_output_languages = len(output_languages)

    # Define colors for each innermost category (lang2: e.g., French Output)
    colors =cm.get_cmap('tab10', num_output_languages)

    # Determine the number of rows and columns for subplots
    num_plots = len(output_languages)
    n_cols = 2 # Number of columns for subplots
    n_rows = (num_plots + n_cols - 1) // n_cols # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    total_group_width = 0.8
    bar_width = total_group_width / num_output_languages # Width of each individual bar within a group

    # Iterate through each outermost category (lang1: e.g., Year) to create a subplot
    for idx, prompt_lang in enumerate(prompt_languages):
        ax = axes[idx] # Get the current subplot axis

        # Get the inner categories for the current prompt_lang (types: e.g., 'No Intervention', 'Hint')
        intervention_types = sorted(list(data[prompt_lang].keys()))
        index = np.arange(len(intervention_types)) # X-axis positions for intervention types

        # Prepare data for plotting in the current subplot
        # This will be a dictionary like {'French Output': [val_no_int, val_hint, ...], ...}
        plot_data_for_subplot = {
            output_lang: [data[prompt_lang][intervention_type].get(output_lang, 0)
                        for intervention_type in intervention_types]
            for output_lang in output_languages
        }

        # Plot each innermost category (lang2: e.g., Product) as a separate set of bars
        for i, output_lang in enumerate(output_languages):
            # Calculate the offset for each group of bars
            offset = (i - (len(output_languages) - 1) / 2) * bar_width
            ax.bar(index + offset, plot_data_for_subplot[output_lang], bar_width,
                label=output_lang, color=colors(i % len(output_languages))) # Use modulo for color cycling

        for i in range(len(intervention_types) - 1):
            ax.axvline(x=i + 0.5, color='gray', linestyle='-', linewidth=1)

        ax.set_xlabel('Intervention Type', fontsize=10)
        ax.set_ylabel(f'Likelihood Score measured in {name}', fontsize=10)
        ax.set_title(f'Output Likelihood with word language {prompt_lang}', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(intervention_types, rotation=45, ha='right', fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.legend(title='Output Language', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide any unused subplots if num_plots is less than n_rows * n_cols
    for i in range(num_plots, n_rows * n_cols):
        fig.delaxes(axes[i])

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    if interactive:
        plt.show()
    else:
        plt.savefig(plt_path)
        plt.close()

def create_histogram_0_to_10(data: dict[str, list[int]], bins: int = 10, title: str = "Histogram of Data (0-10 Range)", xlabel: str = "Value", ylabel: str = "Frequency", interactive=True, plt_path=''):
    """
    Creates a histogram from a dictionary of integer data, where each key represents a series.
    Only data points between 0 and 10 (inclusive) are plotted.
    All series are plotted on a single histogram, distinguished by color.
    This function assumes no NaN values are present in the input data.

    Args:
        data (dict[str, list[int]]): The input dictionary with string keys and lists of integer data.
        bins (int): The number of bins for the histogram within the 0-10 range.
        title (str): The title of the histogram plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """

    all_data_for_plot = [] # To hold all filtered data lists for plotting
    all_labels = []        # To hold labels for each series

    # Define the target range for plotting
    min_plot_value = 0
    max_plot_value = 10

    # Filter data to include only values between 0 and 10
    for key, values in data.items():
        filtered_series_data = [x for x in values if min_plot_value <= x <= max_plot_value]

        if filtered_series_data: # Only add if there's valid data in the filtered series
            all_data_for_plot.append(filtered_series_data)
            all_labels.append(key)
        else:
            print(f"Series '{key}' has no data points within the 0-10 range and will not be plotted.")

    if not all_data_for_plot:
        print("No valid data points to plot from any series within the 0-10 range.")
        return

    # Create bins specifically for the 0-10 range
    # Using np.linspace for evenly spaced bins between 0 and 10
    custom_bins = np.linspace(min_plot_value, max_plot_value, bins + 1)

    print(f"Generated bins for 0-10 range: {custom_bins}")

    # Create the histogram using matplotlib.pyplot
    plt.figure(figsize=(12, 7)) # Set figure size for better readability

    # Plot each series using the custom bins. Matplotlib will automatically assign different colors.
    plt.hist(all_data_for_plot, bins=custom_bins, edgecolor='black', alpha=0.7, label=all_labels)

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add a legend to distinguish the different data series by color
    plt.legend(title="Data Series")

    # Ensure x-axis limits match the 0-10 range
    plt.xlim(min_plot_value, max_plot_value)

    plt.grid(axis='y', alpha=0.75) # Add a grid for better readability
    if interactive:
        plt.show()
    else:
        plt.savefig(plt_path)
        plt.close()

def create_multi_series_histogram(data: dict[str, list[int]], bins: int = 30, title: str = "Histogram of Data Series", xlabel: str = "Value", ylabel: str = "Frequency", interactive=True, plt_path='', file_name = ''):
    """
    Creates a histogram from a dictionary of integer data, where each key represents a series.
    All series are plotted on a single histogram, distinguished by color.
    This function assumes no NaN values are present in the input data.

    Args:
        data (dict[str, list[int]]): The input dictionary with string keys and lists of integer data.
        bins (int): The number of bins for the histogram.
        title (str): The title of the histogram plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """

    all_data_for_plot = [] # To hold all data lists for plotting
    all_labels = []        # To hold labels for each series (keys from the dictionary)

    # Prepare data for plotting
    for key, values in data.items():
        if values: # Only add if the series is not empty
            all_data_for_plot.append(values)
            all_labels.append(key)
        else:
            print(f"Series '{key}' is empty and will not be plotted.")

    if not all_data_for_plot:
        print("No valid data points to plot from any series.")
        return

    # Create the histogram using matplotlib.pyplot
    plt.figure(figsize=(12, 7)) # Set figure size for better readability

    # Plot each series. Matplotlib will automatically assign different colors.
    # 'histtype' can be 'bar', 'barstacked', 'step', 'stepfilled'
    # 'alpha' controls transparency, useful when bars overlap
    plt.hist(all_data_for_plot, bins=bins, edgecolor='black', alpha=0.7, label=all_labels)

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add a legend to distinguish the different data series by color
    plt.legend(title="Data Series")

    plt.grid(axis='y', alpha=0.75) # Add a grid for better readability
    if interactive:
        plt.show()
    else:
        file_path = os.path.join(plt_path, file_name + '_full_hist')
        plt.savefig(file_path)
        plt.close()

    if max(map(max, all_data_for_plot)) > 30:
      new_title = title + ' between 0 and 10'
      file_path = os.path.join(plt_path, file_name + '_10_hist')
      create_histogram_0_to_10(data, title=new_title, interactive=interactive, plt_path=file_path)

    return
