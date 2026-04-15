from datasets import load_dataset
import json
import os
import random
import regex
import torch
from typing import Callable

from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
three_levels_up = current_file_path.parents[2]
sys.path.insert(0, str(three_levels_up))

from circuit_tracer_import import ReplacementModel

# character-checks
def is_english_alphabet_char(char: str) -> bool:
    if len(char) > 1:
        raise AssertionError('Argument should be one character')
    return ('a' <= char <= 'z' or 'A' <= char <= 'Z')

def is_french_alphabet_char(char: str) -> bool:
    if len(char) > 1:
        raise AssertionError('Argument should be one character')
    french_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ"
    return char.isalpha() and char in french_chars

# may not be comprehensive
def is_german_alphabet_char(char: str) -> bool:
    if len(char) > 1:
        raise AssertionError('Argument should be one character')
    
    if not char.isalpha():
        return False
    german_specific_chars = "äöüÄÖÜß"
    if char in german_specific_chars:
        return True
    elif is_english_alphabet_char(char):
        return True
    else:
        return False
    
def is_chinese_alphabet_char(char: str) -> bool:
    if len(char) > 1:
        raise AssertionError('Argument should be one character')
    
    return bool(regex.fullmatch(r'\p{Han}', char))

def is_japanese_alphabet_char(char: str) -> bool:
    if len(char) > 1:
        raise AssertionError('Argument should be one character')
    unicode_value = ord(char)
    # Hiragana range
    if 0x3040 <= unicode_value <= 0x309F:
        return True
    # Katakana range
    elif 0x30A0 <= unicode_value <= 0x30FF:
        return True
    # CJK Unified Ideographs (Kanji) range
    elif 0x4E00 <= unicode_value <= 0x9FFF:
        return True
    return False

def is_spanish_alphabet_char(char: str) -> bool:
    if len(char) > 1:
        raise AssertionError('Argument should be one character')
    
    spanish_chars = "abcdefghijklmnopqrstuvwxyzñáéíóúü"
    char_lower = char.lower()
    return char_lower in spanish_chars

def is_korean_alphabet_char(char: str) -> bool:
    # Hangul Syllables range
    if '\uAC00' <= char <= '\uD7A3':
        return True
    # Hangul Jamo (individual consonants and vowels) range
    if '\u1100' <= char <= '\u11FF':
        return True
    # Hangul Compatibility Jamo range
    if '\u3130' <= char <= '\u318F':
        return True
    return False

alphabet_char = {
    'en': is_english_alphabet_char,
    'fr': is_french_alphabet_char,
    'de': is_german_alphabet_char,
    'es': is_spanish_alphabet_char,
    'zh': is_chinese_alphabet_char,
    'ja': is_japanese_alphabet_char,
    'ko': is_korean_alphabet_char,
}

def english_sentence_clean(sentence: str) -> str:
    words = sentence.split(" ")
    modified_words = []
    for word in words:
        if word == 'i':
            modified_words.append(' I')
        elif word[0] == ',' or word[0] == "'" or word[0] == '?' or word[0] == '.':
            modified_words.append(word)
        else:
            modified_words.append(' ' + word)
    result = "".join(modified_words)
    if result[0] == ' ':
        result = result[1:]
    
    return result

def french_sentence_clean(sentence: str) -> str:
    words = sentence.split(" ")
    modified_words = []
    ends_with_apos = False
    for word in words:
        if ends_with_apos:
            modified_words.append(word)
        elif word[0] == ',' or word[0] == "'" or word[0] == '?' or word[0] == '.':
            modified_words.append(word)
        else:
            modified_words.append(' ' + word)
        
        if word[-1] == "'":
            ends_with_apos = True
        else:
            ends_with_apos = False

    result = "".join(modified_words)
    if result[0] == ' ':
        result = result[1:]
    
    return result

def german_sentence_clean(sentence: str) -> str:
    words = sentence.split(" ")
    modified_words = []
    for word in words:
        if word[0] == ',' or word[0] == "'" or word[0] == '?' or word[0] == '.':
            modified_words.append(word)
        else:
            modified_words.append(' ' + word)

    result = "".join(modified_words)
    if result[0] == ' ':
        result = result[1:]
    
    return result

def chinese_sentence_clean(sentence: str) -> str:
    words = sentence.split(" ")
    modified_words = []
    before_eng_alphabet = False
    for word in words:
        if word[0] == ',' or word[0] == "'" or word[0] == '?' or word[0] == '.':
            modified_words.append(word)
        elif before_eng_alphabet or is_english_alphabet_char(word[0]):
            modified_words.append(' ' + word)
        else:
            modified_words.append(word)
        before_eng_alphabet = is_english_alphabet_char(word[-1])

    result = "".join(modified_words)
    if result[0] == ' ':
        result = result[1:]
    
    return result

def sentences_clean(sentences: list[str], function: Callable) -> list[str]:
    cleaned = []
    for sentence in sentences:
        cleaned.append(function(sentence))
    return cleaned

def is_valid_word(word: str, function: Callable) -> bool:
    if function(word[0]):
        return True
    elif word[0] == ' ' and len(word) > 1:
        return function(word[1])
    else:
        return False

# min_tokens is for the minimum length of the original sentence, 
# max_tokens is for the maximum length for the resulting incomplete sentences
# if the training takes too long, lower max_tokens
def filter_sentences(
        sentences: list[str], 
        function: Callable, 
        model: ReplacementModel, 
        num_sentences: int=100, 
        min_tokens: int=6, 
        max_tokens: int=20, 
        random_seed: int = 42
    ) -> list[str]:
    MAX_ITERATIONS = 20
    random.seed(random_seed)
    filtered = []
    for sentence in sentences:
        if len(filtered) >= num_sentences:
            break
        tokenized = model.tokenizer.encode(sentence)
        n = len(tokenized)
        if n <= min_tokens:
            continue
        random_token = random.randint(3, min(n-2, max_tokens))
        next_token = model.tokenizer.decode(tokenized[random_token])
        iterations = 0
        while not (is_valid_word(next_token, function)):
            if iterations > MAX_ITERATIONS:
                print(sentence)
                raise TypeError('No character of the specified language found')
            random_token = random.randint(3, min(n-2, max_tokens))
            next_token = model.tokenizer.decode(tokenized[random_token])
            iterations += 1
        decoded = model.tokenizer.decode(tokenized[1:random_token])
        filtered.append(decoded)
    if len(filtered) < num_sentences:
        raise ValueError('Not enough good sentences')
    return filtered
