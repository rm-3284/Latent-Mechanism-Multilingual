hf_model_names = {
    'gemma-2-2b': 'google/gemma-2-2b',
    'qwen3-4b': 'Qwen/Qwen3-4B',
    'gemma-3-4b-it': 'google/gemma-3-4b-it',
    'gemma-2-2b-pt': 'google/gemma-2-2b',
}
hf_transcoder_names = {
    'gemma-2-2b': 'mwhanna/gemma-scope-transcoders',
    'qwen3-4b': 'mwhanna/qwen3-4b-transcoders',
    'gemma-3-4b-it': 'google/gemma-scope-2-4b-it',
    'gemma-2-2b-pt': 'google/gemma-scope-2b-pt-transcoders',
}
base_neuronpeida_url = 'https://www.neuronpedia.org/api/feature/'
neuronpedia_urls = {
    'gemma-2-2b': base_neuronpeida_url + 'gemma-2-2b/{layer}-clt-hp/{feature_idx}',
    'qwen3-4b': base_neuronpeida_url + 'qwen3-4b/{layer}-transcoder-hp/{feature_idx}',
    'gemma-3-4b-it': base_neuronpeida_url + 'gemma-3-4b-it/{layer}-gemmascope-2-transcoder-262k/{feature_idx}',
    'gemma-2-2b-pt': base_neuronpeida_url + 'gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature_idx}',
}
layer_num = {
    'gemma-2-2b': 26,
    'qwen3-4b': 36,
    'gemma-3-4b-it': 34,
    'gemma-2-2b-pt': 26,
}
n_features = {
    'gemma-2-2b': 16384,
    'qwen3-4b': 163840,
    'gemma-3-4b-it': 262144,
    'gemma-2-2b-pt': 16384,
}
use_bos = {
    'gemma-2-2b': True,
    'qwen3-4b': False,
    'gemma-3-4b-it': True,
    'gemma-2-2b-pt': True,
}
