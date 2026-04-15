import argparse
import json
import nnsight
import os
import torch
import torch.nn.functional as F

from ablation_amplification_intervention import (
  activation_dict,
  ablation_and_amplification,
  combine_except_one,
  description_based_features,
  direction_ablation_layer_determine,
  direction_ablation_helper,
  freq_based_features,
  mean_value_based_features,
  model_intervention,
  model_run,
)
from circuit_tracer_import import ReplacementModel
from device_setup import device
from direction_ablation import (
  interventions_to_dict,
  interventions_to_dict_everything_ablation,
  project_orthogonally,
)
from intervention import ablation, amplification, get_top_outputs
from template import lang_to_flores_key
from models import hf_model_names, hf_transcoder_names, layer_num

## data used
base_prompts = {
    "en": "The {category} are: {choices}",
    "de": "Die {category} sind: {choices}",
    "fr": "Les {category} sont : {choices}",
    "es": "Los {category} son: {choices}",
    "zh": "{category}是：{choices}",
    "ja": "{category}は次の通りです：{choices}",
    "ko": "{category}는 다음과 같습니다: {choices}"
}

categories = {
    "months": {
      "en": "months of the year",
      "de": "Monate des Jahres",
      "fr": "mois de l'année",
      "es": "meses del año",
      "zh": "一年中的月份",
      "ja": "一年の月",
      "ko": "1년의 열두 달"
    },
    "numbers": {
      "en": "numbers from one to ten",
      "de": "Zahlen von eins bis zehn",
      "fr": "nombres de un à dix",
      "es": "números del uno al diez",
      "zh": "从一到十的数字",
      "ja": "一から十までの数字",
      "ko": "1부터 10까지의 숫자"
    },
    "days_of_week": {
      "en": "days of the week",
      "de": "Wochentage",
      "fr": "jours de la semaine",
      "es": "días de la semana",
      "zh": "一星期中的日子",
      "ja": "曜日",
      "ko": "요일"
    },
    "four_seasons": {
      "en": "four seasons",
      "de": "vier Jahreszeiten",
      "fr": "quatre saisons",
      "es": "cuatro estaciones",
      "zh": "四季",
      "ja": "四季",
      "ko": "사계절"
    },
    "times_of_day": {
      "en": "times of day",
      "de": "Tageszeiten",
      "fr": "moments de la journée",
      "es": "momentos del día",
      "zh": "一天的时段",
      "ja": "一日の時間帯",
      "ko": "하루의 시간대"
    },
    "cardinal_directions": {
      "en": "cardinal directions",
      "de": "Himmelsrichtungen",
      "fr": "points cardinaux",
      "es": "puntos cardinales",
      "zh": "基本方位",
      "ja": "基本方位",
      "ko": "방위"
    },
    "primary_colors_of_light": {
      "en": "primary colors of light",
      "de": "Primärfarben des Lichts",
      "fr": "couleurs primaires de la lumière",
      "es": "colores primarios de la luz",
      "zh": "光的三原色",
      "ja": "光の三原色",
      "ko": "빛의 삼원색"
    }
  }

options = {
  "months": {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "zh": ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"],
    "ja": ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"],
    "ko": ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
  },
  "numbers": {
    "en": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "de": ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn"],
    "fr": ["un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix"],
    "es": ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"],
    "zh": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    "ja": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    "ko": ["일", "이", "삼", "사", "오", "육", "칠", "팔", "구", "십"]
  },
  "days_of_week": {
    "en": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "de": ["Sonntag", "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag"],
    "fr": ["dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"],
    "es": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
    "zh": ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"],
    "ja": ["日曜日", "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日"],
    "ko": ["일요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일"]
  },
  "four_seasons": {
    "en": ["spring", "summer", "fall", "winter"],
    "de": ["Frühling", "Sommer", "Herbst", "Winter"],
    "fr": ["printemps", "été", "automne", "hiver"],
    "es": ["primavera", "verano", "otoño", "invierno"],
    "zh": ["春", "夏", "秋", "冬"],
    "ja": ["春", "夏", "秋", "冬"],
    "ko": ["봄", "여름", "가을", "겨울"]
  },
  "times_of_day": {
    "en": ["morning", "afternoon", "evening", "night"],
    "de": ["Morgen", "Nachmittag", "Abend", "Nacht"],
    "fr": ["matin", "après-midi", "soir", "nuit"],
    "es": ["mañana", "tarde", "tarde", "noche"],
    "zh": ["早上", "下午", "晚上", "夜里"],
    "ja": ["朝", "午後", "晩", "夜"],
    "ko": ["아침", "오후", "저녁", "밤"]
  },
  "cardinal_directions": {
    "en": ["North", "South", "East", "West"],
    "de": ["Norden", "Süden", "Osten", "Westen"],
    "fr": ["nord", "sud", "est", "ouest"],
    "es": ["norte", "sur", "este", "oeste"],
    "zh": ["北", "南", "东", "西"],
    "ja": ["北", "南", "東", "西"],
    "ko": ["북", "남", "동", "서"]
  },
  "primary_colors_of_light": {
    "en": ["red", "green", "blue"],
    "de": ["Rot", "Grün", "Blau"],
    "fr": ["rouge", "vert", "bleu"],
    "es": ["rojo", "verde", "azul"],
    "zh": ["红", "绿", "蓝"],
    "ja": ["赤", "緑", "青"],
    "ko": ["빨강", "초록", "파랑"]
  }
}

number_of_choices_to_display = {
    "months": 4,
    "numbers": 4,
    "days_of_week": 3,
    "four_seasons": 2,
    "times_of_day": 2,
    "cardinal_directions": 2,
    "primary_colors_of_light": 1
}


# Compute the log-probability of a full string as a continuation from a prompt
def get_logprob_of_string(prompt: str, target: str | list[str], model: ReplacementModel) -> float:
  # If target is a list, join with comma
  if isinstance(target, list):
    target_str = ", ".join(target)
  else:
    target_str = target
  prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
  target_ids = model.tokenizer.encode(target_str, add_special_tokens=False)
  input_ids = prompt_ids + target_ids
  input = model.tokenizer.decode(input_ids)
  _, logits = model_run(input, model)
  log_probs = F.log_softmax(logits[0, len(prompt_ids)-1:len(input_ids)-1], dim=-1)
  total_logprob = 0.0
  for i, tid in enumerate(target_ids):
    total_logprob += log_probs[i, tid].item()
  return total_logprob

# layer, pos, feature_idx, ablation_value with pos=-1 into pos=[start_pos, last_pos]
def transform_intervention(intervention: list[tuple[int, int, int, float]], start_pos: int, last_pos: int) -> list[tuple[int, int, int, float]]:
    transformed = []
    for layer, pos, feature_idx, ablation_value in intervention:
        if pos == -1:
            for p in range(start_pos, last_pos+1):
                transformed.append((layer, p, feature_idx, ablation_value))
    return transformed

# log-prob with intervention
def get_logprob_with_intervention(prompt: str, target: str, model: ReplacementModel, intervention: list[tuple[int, int, int, float]]) -> float:
  # If target is a list, join with comma
  if isinstance(target, list):
    target_str = ", ".join(target)
  else:
    target_str = target

  prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
  target_ids = model.tokenizer.encode(target_str, add_special_tokens=False)
  input_ids = prompt_ids + target_ids
  full_input = model.tokenizer.decode(input_ids)
  start_pos = len(prompt_ids)
  last_pos = len(input_ids)-1
  transformed_intervention = transform_intervention(intervention, start_pos, last_pos)
  logits, _ = model.feature_intervention(full_input, transformed_intervention, return_activations=False)
  log_probs = F.log_softmax(logits[0, len(prompt_ids)-1:len(input_ids)-1], dim=-1)
  total_logprob = 0.0
  for i, tid in enumerate(target_ids):
    total_logprob += log_probs[i, tid].item()
  return total_logprob

# For each language, compute the log-prob of each candidate string as a continuation
def get_logprobs_for_candidates(prompt: str, candidates: dict[str, list[str]], model: ReplacementModel) -> dict[str, dict[str, float]]:
    result = dict()
    for lang, candidate_str in candidates.items():
        result[lang] = dict()
        logprob = get_logprob_of_string(prompt, candidate_str, model)
        result[lang][candidate_str] = logprob
    return result


# For each intervention, compute log-prob of each candidate string as a continuation from the intervened prompt
def feature_interventions_logprob(prompt: str, model: ReplacementModel, ans: dict[str, str], intervention: dict[str, list[tuple[int, int, int, float]]], langs: list[str]):
    results = dict()
    for intervened_lang in langs:
        results[intervened_lang] = dict()
        # Get ablated logits for the prompt using feature_intervention
        for lang in langs:
          results[intervened_lang][lang] = dict()
          candidate = ans[lang]
          total_logprob = get_logprob_with_intervention(prompt, candidate, model, intervention[intervened_lang])
          results[intervened_lang][lang][candidate] = total_logprob
    return results

# direction_ablation helper
def run_ablation_experiment(model, prompt, target, directions_map):
    """
    Performs direction ablation at each position to see how it affects 
    the prediction of the next token in the target sequence.
    """
    #device = next(model.parameters()).devie
    # 1. Tokenize the full sequence
    # We concatenate prompt + target. 
    # Example: "The capital of France is" + " Paris"
    full_prompt = prompt + target
    tokens = model.tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    
    # Identify where the target begins in the full sequence
    prompt_tokens = model.tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = prompt_tokens.shape[1]
    total_len = tokens.shape[1]

    with model.trace(tokens):
        # Determine architecture-specific layer path
        if hasattr(model, "transformer"):
            layers = model.transformer.h
        elif hasattr(model, "model"):
            layers = model.model.layers
        else:
            layers = [m for m in model.modules() if isinstance(m, torch.nn.ModuleList)]

        for i, layer in enumerate(layers):
            if i in directions_map:
                dirs = directions_map[i].to(device)
                
                # hidden_states shape: [batch, seq_len, d_model]
                hidden_states = layer.output[0]
                
                # Apply ablation to ALL positions or just the target positions?
                # Usually, we ablate from prompt_len - 1 (the pos that predicts target[0])
                # up to the second to last token (which predicts target[-1]).
                ablated_states = project_orthogonally(hidden_states, dirs)
                
                # Apply the change
                layer_output = (ablated_states, ) + layer.output[1:]
                layer.output = layer_output

        # Save logits
        # shape: [batch, seq_len, vocab_size]
        output_logits = model.lm_head.output.save()

    # 2. Extract only the logits that predict the target tokens
    # Logit at index [prompt_len - 1] predicts target[0]
    # Logit at index [total_len - 2] predicts target[-1]
    target_logits = output_logits[0, prompt_len-1 : total_len-1, :]
    
    return target_logits

# For direction ablation, also compute log-prob for each candidate string as a continuation
def direction_ablate_logprob(prompt: str, model: ReplacementModel, ans, interventions, langs, nnsight_model):
    results = dict()
    for intervention_lang in langs:
      results[intervention_lang] = dict()
      # Use nnsight_model to apply the ablation intervention for this language
      intervention = interventions[intervention_lang]
      for lang in langs:
            results[intervention_lang][lang] = dict()
            candidate = ans[lang]
            candidate_str = ", ".join(candidate) if isinstance(candidate, list) else candidate
            target_ids = model.tokenizer.encode(candidate_str, add_special_tokens=False)
            logits = run_ablation_experiment(nnsight_model, prompt, candidate_str, intervention)
            log_probs = F.log_softmax(logits, dim=-1)
            if len(target_ids) != log_probs.shape[0]:
                print(f"[WARNING] target_ids length ({len(target_ids)}) != log_probs rows ({log_probs.shape[0]}) for candidate '{candidate_str}' with prompt '{prompt}'. Skipping.")
                continue
            total_logprob = 0.0
            for i, tid in enumerate(target_ids):
                total_logprob += log_probs[i, tid].item()
            results[intervention_lang][lang][candidate_str] = total_logprob
    return results


# For ablation+amplification, compute log-prob of each candidate string as a continuation from the prompt under the intervention
def feature_ablation_and_amplification_logprob(prompt: str, model: ReplacementModel, ans: dict[str, str], ablation, amplification, langs):
  results = dict()
  for ablation_lang in langs:
    results[ablation_lang] = dict()
    for amplification_lang in langs:
      results[ablation_lang][amplification_lang] = dict()
      interventions = ablation_and_amplification(ablation[ablation_lang], amplification[amplification_lang])
      for lang in langs:
        results[ablation_lang][amplification_lang][lang] = dict()
        candidate = ans[lang]
        total_logprob = get_logprob_with_intervention(prompt, candidate, model, interventions)
        results[ablation_lang][amplification_lang][lang][candidate] = total_logprob
  return results

def parse_args():
  parser = argparse.ArgumentParser(
    description="Prompt language",
    formatter_class=argparse.RawTextHelpFormatter
  )

  parser.add_argument(
    '--lang',
    '-l',
    type=str,
    default=None,
    help='Prompt language',
  )
  parser.add_argument(
    '--list_lang',
    type=str,
    default=None,
    help='Languages to list features for, separated by comma. If provided, --lang is ignored.',
  )
  parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='gemma-2-2b',
    choices=hf_model_names.keys(),
    help='Model to use for the experiment',
  )

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()

  # load the model
  model_name = args.model
  transcoder_name = hf_transcoder_names.get(model_name, "gemma")
  model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

  nnsight_model = nnsight.LanguageModel(hf_model_names[model_name], device_map=device)

  # relevant directories
  current_file_path = __file__
  current_directory = os.path.dirname(current_file_path)
  absolute_directory = os.path.abspath(current_directory)
  data_directory = os.path.join(os.path.dirname(absolute_directory), "data")
  flores_directory = os.path.join(data_directory, "flores_features")#, model_name)
  lang_specific_directory = os.path.join(data_directory, "language_specific_features")#, model_name)
  multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features")#, model_name)
  amplification_values_directory = os.path.join(data_directory, "amplification_values")#, model_name)

  langs = list(lang_to_flores_key.keys())

  # get the features + amplification values
  desc_features = description_based_features(flores_directory, langs, 0.1)
  val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
  freq_features = freq_based_features(lang_specific_directory, langs)

  desc_interventions = activation_dict(desc_features, amplification_values_directory, langs)
  val_interventions = activation_dict(val_features, amplification_values_directory, langs)
  freq_interventions = activation_dict(freq_features, amplification_values_directory, langs)

  desc_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
  desc_amplifications = {'normal': dict(), 'everything': dict()}
  val_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
  val_amplifications = {'normal': dict(), 'everything': dict()}
  freq_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
  freq_amplifications = {'normal': dict(), 'everything': dict()}
  for lang in langs:
    desc_ablations['feature'][lang] = ablation(desc_interventions, lang)
    desc_amplifications['normal'][lang] = amplification(desc_interventions, lang)
    val_ablations['feature'][lang] = ablation(val_interventions, lang)
    val_amplifications['normal'][lang] = amplification(val_interventions, lang)
    freq_ablations['feature'][lang] = ablation(freq_interventions, lang)
    freq_amplifications['normal'][lang] = amplification(freq_interventions, lang)

  for lang in langs:
    desc_ablations['feature_everything'][lang] = combine_except_one(desc_ablations['feature'], lang)
    desc_amplifications['everything'][lang] = combine_except_one(desc_amplifications['normal'], lang)
    val_ablations['feature_everything'][lang] = combine_except_one(val_ablations['feature'], lang)
    val_amplifications['everything'][lang] = combine_except_one(val_amplifications['normal'], lang)
    freq_ablations['feature_everything'][lang] = combine_except_one(freq_ablations['feature'], lang)
    freq_amplifications['everything'][lang] = combine_except_one(freq_amplifications['normal'], lang)

  for lang in langs:
    desc_ablations['direction-ablation'][lang] = interventions_to_dict(desc_interventions, lang, model)
    desc_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(desc_interventions, lang, model)
    val_ablations['direction-ablation'][lang] = interventions_to_dict(val_interventions, lang, model)
    val_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(val_interventions, lang, model)
    freq_ablations['direction-ablation'][lang] = interventions_to_dict(freq_interventions, lang, model)
    freq_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(freq_interventions, lang, model)

  one_layer_ablation = {'desc': dict(), 'val': dict(), 'freq': dict()}
  for lang in langs:
    one_layer_ablation['desc'][lang] = direction_ablation_layer_determine(desc_interventions, lang, num_layers=layer_num[model_name])
    one_layer_ablation['val'][lang] = direction_ablation_layer_determine(val_interventions, lang, num_layers=layer_num[model_name])
    one_layer_ablation['freq'][lang] = direction_ablation_layer_determine(freq_interventions, lang, num_layers=layer_num[model_name])

  # ablation + amplification experiments
  output_dir = os.path.join(data_directory, "interventions_multiple_words", model_name)
  for prompt_lang in langs:
    if args.lang is not None and prompt_lang != args.lang:
      continue

    lang_out_dir = os.path.join(output_dir, prompt_lang)
    base = base_prompts[prompt_lang]

    for list_lang in langs:
      if args.list_lang is not None and list_lang not in args.list_lang.split(","):
        continue
      list_lang_out_dir = os.path.join(lang_out_dir, list_lang)
      os.makedirs(list_lang_out_dir, exist_ok=True)

      desc_based = {
        'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {},
        'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {},
        'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {},
        'feature-intervention': {}, 'one-layer direction intervention': {}, }
      val_based = {
        'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {},
        'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {},
        'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {},
        'feature-intervention': {}, 'one-layer direction intervention': {}, }
      freq_based = {
        'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {},
        'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {},
        'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {},
        'feature-intervention': {}, 'one-layer direction intervention': {}, }


      for category_key, category_by_lang in categories.items():
        options_key = category_key
        category_text = category_by_lang[prompt_lang]
        n_display = number_of_choices_to_display[category_key]
        display_choices = options[options_key][list_lang][:n_display]
        predict_choices = options[options_key][list_lang][n_display:]
        if list_lang != "ja" and list_lang != 'zh':
          fill_in = ", ".join(display_choices) + ","
        else:
          fill_in = ",".join(display_choices) + ","
        prompt = base.format(category=category_text, choices=fill_in)
        # For each language, get the remaining options to predict
        ans = {lang: options[options_key][lang][n_display:] for lang in langs}
        # turn this into a str
        for lang in langs:
          if lang != "ja" and lang != 'zh':
             expected = " " + ", ".join(ans[lang])
          else:
             expected = ",".join(ans[lang])
          ans[lang] = expected

        print(f"Prompt: {prompt}")
        print(f"Answers: {ans}")

        # calculate the direction ablations
        for lang in langs:
          layer, features = one_layer_ablation['desc'][lang]
          interventions = direction_ablation_helper(model, layer, features, prompt)
          desc_ablations['one-layer-direction'][lang] = interventions
        for lang in langs:
          desc_ablations['one-layer-direction_everything'][lang] = combine_except_one(desc_ablations['one-layer-direction'], lang)

        for lang in langs:
          layer, features = one_layer_ablation['val'][lang]
          interventions = direction_ablation_helper(model, layer, features, prompt)
          val_ablations['one-layer-direction'][lang] = interventions
        for lang in langs:
          val_ablations['one-layer-direction_everything'][lang] = combine_except_one(val_ablations['one-layer-direction'], lang)

        for lang in langs:
          layer, features = one_layer_ablation['freq'][lang]
          interventions = direction_ablation_helper(model, layer, features, prompt)
          freq_ablations['one-layer-direction'][lang] = interventions
        for lang in langs:
          freq_ablations['one-layer-direction_everything'][lang] = combine_except_one(freq_ablations['one-layer-direction'], lang)


        # original: compute log-prob for each remaining option as a continuation
        print("Calculating original log-probs...")
        logprobs = get_logprobs_for_candidates(prompt, ans, model)
        desc_based['original'][prompt] = {'logprobs': logprobs}
        val_based['original'][prompt] = {'logprobs': logprobs}
        freq_based['original'][prompt] = {'logprobs': logprobs}


        # distractor ablation
        print("Calculating distractor ablation log-probs...")
        desc_based['distractor ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, desc_ablations['feature'], langs)
        val_based['distractor ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, val_ablations['feature'], langs)
        freq_based['distractor ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, freq_ablations['feature'], langs)

        # ablation
        print("Calculating ablation log-probs...")
        desc_based['ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, desc_ablations['feature_everything'], langs)
        val_based['ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, val_ablations['feature_everything'], langs)
        freq_based['ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, freq_ablations['feature_everything'], langs)

        # distractor one-layer direction
        print("Calculating distractor one-layer direction ablation log-probs...")
        desc_based['distractor one-layer direction ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, desc_ablations['one-layer-direction'], langs)
        val_based['distractor one-layer direction ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, val_ablations['one-layer-direction'], langs)
        freq_based['distractor one-layer direction ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, freq_ablations['one-layer-direction'], langs)

        # one-layer direction ablation
        print("Calculating one-layer direction ablation log-probs...")
        desc_based['one-layer direction ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, desc_ablations['one-layer-direction_everything'], langs)
        val_based['one-layer direction ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, val_ablations['one-layer-direction_everything'], langs)
        freq_based['one-layer direction ablation'][prompt] = feature_interventions_logprob(prompt, model, ans, freq_ablations['one-layer-direction_everything'], langs)

        # distractor multi-layer direction ablation
        print("Calculating distractor multi-layer direction ablation log-probs...")
        desc_based['distractor multi-layer direction ablation'][prompt] = direction_ablate_logprob(prompt, model, ans, desc_ablations['direction-ablation'], langs, nnsight_model)
        val_based['distractor multi-layer direction ablation'][prompt] = direction_ablate_logprob(prompt, model, ans, val_ablations['direction-ablation'], langs, nnsight_model)
        freq_based['distractor multi-layer direction ablation'][prompt] = direction_ablate_logprob(prompt, model, ans, freq_ablations['direction-ablation'], langs, nnsight_model)

        # multi-layer direction ablation
        print("Calculating multi-layer direction ablation log-probs...")
        desc_based['multi-layer direction ablation'][prompt] = direction_ablate_logprob(prompt, model, ans, desc_ablations['direction-ablation-everything'], langs, nnsight_model)
        val_based['multi-layer direction ablation'][prompt] = direction_ablate_logprob(prompt, model, ans, val_ablations['direction-ablation-everything'], langs, nnsight_model)
        freq_based['multi-layer direction ablation'][prompt] = direction_ablate_logprob(prompt, model, ans, freq_ablations['direction-ablation-everything'], langs, nnsight_model)

        # amplification
        print("Calculating amplification log-probs...")
        desc_based['amplification'][prompt] = feature_interventions_logprob(prompt, model, ans, desc_amplifications['everything'], langs)
        val_based['amplification'][prompt] = feature_interventions_logprob(prompt, model, ans, val_amplifications['everything'], langs)
        freq_based['amplification'][prompt] = feature_interventions_logprob(prompt, model, ans, freq_amplifications['everything'], langs)

        # non-distractor amplification
        print("Calculating non-distractor amplification log-probs...")
        desc_based['non-distractor amplification'][prompt] = feature_interventions_logprob(prompt, model, ans, desc_amplifications['normal'], langs)
        val_based['non-distractor amplification'][prompt] = feature_interventions_logprob(prompt, model, ans, val_amplifications['normal'], langs)
        freq_based['non-distractor amplification'][prompt] = feature_interventions_logprob(prompt, model, ans, freq_amplifications['normal'], langs)

        # feature-intervention
        print("Calculating feature-intervention log-probs...")
        desc_based['feature-intervention'][prompt] = feature_ablation_and_amplification_logprob(prompt, model, ans, desc_ablations['feature'], desc_amplifications['normal'], langs)
        val_based['feature-intervention'][prompt] = feature_ablation_and_amplification_logprob(prompt, model, ans, val_ablations['feature'], val_amplifications['normal'], langs)
        freq_based['feature-intervention'][prompt] = feature_ablation_and_amplification_logprob(prompt, model, ans, freq_ablations['feature'], freq_amplifications['normal'], langs)

        # one-layer direction intervention
        print("Calculating one-layer direction intervention log-probs...")
        desc_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification_logprob(prompt, model, ans, desc_ablations['one-layer-direction'], desc_amplifications['normal'], langs)
        val_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification_logprob(prompt, model, ans, val_ablations['one-layer-direction'], val_amplifications['normal'], langs)
        freq_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification_logprob(prompt, model, ans, freq_ablations['one-layer-direction'], freq_amplifications['normal'], langs)

      filename = "interventions_and_results_description.json"
      with open(os.path.join(list_lang_out_dir, filename), 'w') as f:
        json.dump(desc_based, f, indent=4)

      filename = "interventions_and_results_value.json"
      with open(os.path.join(list_lang_out_dir, filename), 'w') as f:
        json.dump(val_based, f, indent=4)

      filename = "interventions_and_results_frequency.json"
      with open(os.path.join(list_lang_out_dir, filename), 'w') as f:
        json.dump(freq_based, f, indent=4)
