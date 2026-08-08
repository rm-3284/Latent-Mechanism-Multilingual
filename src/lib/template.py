base_strings = {
    'en': 'The opposite of "{adj}" is "',
    'fr': 'Le contraire de "{adj}" est "',
    'es': 'Lo opuesto de "{adj}" es "',
    'de': 'Das Gegenteil von "{adj}" ist "',
    'zh':'"{adj}"的反义词是"',
    'ja':'「{adj}」の反対は「',
    'ko':'"{adj}"의 반대말은 "',
}
langs = {'en', 'fr', 'de', 'zh', 'ja'}
langs_big = {'en', 'fr', 'de', 'zh', 'ja', 'es', 'ko'}
lang_dict = {
    'en': {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German', 
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',},
    'ja': {'en': '英語', 'fr': 'フランス語', 'es': 'スペイン語', 'de': 'ドイツ語',
           'zh': '中国語', 'ja': '日本語', 'ko': '韓国語',},
    'fr': {'en': 'anglais', 'fr': 'français', 'es': 'espagnol', 
           'de': 'allemand', 'zh': 'chinois', 'ja': 'japonais', 'ko': 'coréen',},
    'es': {'en': 'inglés', 'fr': 'francés', 'es': 'español', 'de': 'alemán',
           'zh': 'chino', 'ja': 'japonés', 'ko': 'coreano'},
    'de': {'en': 'Englisch', 'fr': 'Französisch', 'es': 'Spanisch', 
           'de': 'Deutsch', 'zh': 'Chinesisch', 'ja': 'Japanisch', 'ko': 'Koreanisch'},
    'zh': {'en': '英语', 'fr': '法语', 'es': '西班牙语', 'de': '德语', 
           'zh': '中文', 'ja': '日文', 'ko': '韩文'},
    'ko': {'en': '영어', 'fr': '프랑스어', 'es': '스페인어', 'de': '독일어', 
           'zh': '중국어', 'ja': '일본어', 'ko': '한국어'},
    }

identifiers = {
    'fr': ['French', 'french', 'France', 'france'],
    'de': ['German', 'german', 'Germany', 'germany'],
    'ja': ['Japanese', 'japanese', 'Japan', 'japan'],
    'zh': ['Chinese', 'chinese', 'China', 'china'],
    'en': ['English', 'english', 'England', 'england'],
    'ar': ['Arabic', 'arabic', 'Arab', 'arab'],
    'es': ['Spanish', 'spanish', 'Spain', 'spain'],
    'ko': ['Korean', 'korean', 'Korea', 'korea'],
    'pt': ['Portuguese', 'portuguese', 'Portugal', 'portugal'],
    'th': ['Thai', 'thai', 'Thailand', 'thailand'],
    'vi': ['Vietnamese', 'vietnamese', 'Vietnam', 'vietnam'],
    'bg': ['Bulgarian', 'bulgarian', 'Bulgaria', 'bulgaria'],
    'ru': ['Russian', 'russian', 'Russia', 'russia'],
    'tr': ['Turkish', 'turkish', 'Turkey', 'turkey'],
}

lang_to_flores_key ={
    'en': 'eng_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'es': 'spa_Latn',
    'zh': 'cmn_Hans',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
}
