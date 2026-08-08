"""
Small / Big, Heavy / Light, Hot / Cold, New / Old, Soft / Hard, Near / Far, Dark / Light,
Young / Old, Quiet / Loud (Noisy), True / False, Active / Passive, Fast / Slow,
"""

adj_dict = {
    'en': 'small', 'fr': 'petit', 'es': 'pequeño', 'de': 'klein', 
    'zh': '小', 'ja': '小さい', 'ko': '작다',
}
ans_dict = {
    'en': ['big', 'large'], 'fr': ['grand', 'gros', 'énorme'], 'es': ['grande', 'enorme', 'gigante'], 
    'de': ['groß'], 'zh': ['大'], 'ja': ['大きい'], 'ko': ['크다', '큰']
}

adj_dict2 = {
    'en': 'heavy', 'fr': 'lourd', 'es': 'pesado', 'de': 'schwer', 
    'zh': '重', 'ja': '重い', 'ko': '무거운',
}
ans_dict2 = {
    'en': ['light'], 'fr': ['léger'], 'es': ['ligero'], 
    'de': ['leicht', 'einfach'], 'zh': ['轻', '軽', '輕'], 'ja': ['軽い'], 'ko': ['가벼운']
}

adj_dict3 = {
    'en': 'hot', 'fr': 'chaud', 'es': 'caliente', 'de': 'heiß', 
    'zh': '热', 'ja': '熱い', 'ko': '뜨거운',
}
ans_dict3 = {
    'en': ['cold'], 'fr': ['froid'], 'es': ['frío', 'helado', 'fresco'], 'de': ['kalt'],
    'zh': ['冷', '凉'], 'ja': ['冷たい'], 'ko': ['차가운'], 
}

adj_dict4 = {
    'en': 'new', 'fr': 'nouveau', 'es': 'nuevo', 'de': 'neu',
    'zh': '新', 'ja': '新しい', 'ko': '새롭다',
}
ans_dict4 = {
    'en': ['old'], 'fr': ['vieux', 'ancien'], 'es': ['viejo', 'antiguo'], 'de': ['alt'],
    'zh': ['旧', '老'], 'ja': ['古い'], 'ko': ['오래되다', '낡다'],
}

adj_dict5 = {
    'en': 'soft', 'fr': 'doux', 'es': 'suave', 'de': 'weich',
    'zh': '软', 'ja': '柔らかい', 'ko': '부드럽다',
}
ans_dict5 = {
    'en': ['hard'], 'fr': ['dur', 'amer', 'acide', 'aigre', 'fort', 'violent', 'brutal', 'raide'], 'es': ['duro', 'fuerte', 'áspero', 'rugoso', 'brusco', 'violento', 'agrio'], 'de': ['hart'],
    'zh': ['硬', '强'], 'ja': ['硬い'], 'ko': ['딱딱하다', '단단하다'],
}

adj_dict6 = {
    'en': 'near', 'fr': 'près', 'es': 'cerca', 'de': 'nah', 
    'zh': '近', 'ja': '近い', 'ko': '가깝다'
}
ans_dict6 = {
    'en': ['far'], 'fr': ['loin', 'éloigné'], 'es': ['lejos'], 'de': ['weit', 'fern'], 
    'zh': ['远', '遠'], 'ja': ['遠い'], 'ko': ['멀다', '먼'],
}

adj_dict7 = {
    'en': 'dark', 'fr': 'foncé', 'es': 'oscuro', 'de': 'dunkel',
    'zh': '暗', 'ja': '暗い', 'ko': '어둡다',
}
ans_dict7 = {
    'en': ['light', 'bright'], 'fr': ['clair', 'pâle', 'brillant'], 'es': ['claro', 'luminoso', 'brillante'],
    'de': ['hell'], 'zh': ['亮', '明', '公开'], 'ja': ['明るい'], 'ko': ['밝다'],
}

adj_dict8 = {
    'en': 'young', 'fr': 'jeune', 'es': 'joven', 'de': 'jung',
    'zh': '年轻', 'ja': '若い', 'ko': '젊다',
}
ans_dict8 = {
    'en': ['old'], 'fr': ['vieux', 'vielle', 'âgé', 'ancien'], 'es': ['viejo', 'mayor'], 'de': ['alt'],
    'zh': ['老', '年老', '年长'], 'ja': ['古い', '年を取った', '高齢の', '老いた'], 'ko': ['늙다', '오래되다'], 
}

adj_dict9 = {
    'en': 'quiet', 'fr': 'silencieux', 'es': 'tranquilo', 'de': 'leise',
    'zh': '安静', 'ja': '静かな', 'ko': '조용하다',
}
ans_dict9 = {
    'en': ['loud', 'noisy'], 'fr': ['bruyant', 'fort', 'tapageur'], 'es': ['ruidoso', 'nervioso', 'agitado', 'inquieto'], 'de': ['laut'],
    'zh': ['吵闹', '大声', '嘈杂', '热闹', '喧闹'], 'ja': ['うるさい', '騒がしい', '賑やかな'], 'ko': ['시끄럽다', '떠들썩하다', '활기차다'],
}

adj_dict10 = {
    'en': 'true', 'fr': 'vrai', 'es': 'verdadero', 'de': 'wahr',
    'zh': '真', 'ja': '本当の', 'ko': '진실하다',
}
ans_dict10 = {
    'en': ['false'], 'fr': ['faux'], 'es': ['falso'], 'de': ['falsch'],
    'zh': ['假'], 'ja': ['偽の', '嘘の', '間違った'], 'ko': ['거짓되다', '가짜이다', '불성실하다'],
}

adj_dict11 = {
    'en': 'active', 'fr': 'actif', 'es': 'activo', 'de': 'aktiv',
    'zh': '主动', 'ja': '能動的な', 'ko': '능동적이다',
}
ans_dict11 = {
    'en': ['passive', 'inactive', 'dormant'], 'fr': ['passif', 'inactif', 'sédentaire', 'paresseux'], 'es': ['pasivo', 'inactivo', 'sedentario', 'ocioso', 'latente'], 'de': ['passiv', 'inaktiv'],
    'zh': ['被动'], 'ja': ['受動的な', '消極的な', '怠惰な'], 'ko': ['수동적이다', '소극적이다'],
}

adj_dict12= {
    'en': 'fast', 'fr': 'rapide', 'es': 'rápido', 'de': 'schnell',
    'zh': '快', 'ja': '速い', 'ko': '빠르다',
}
ans_dict12 = {
    'en': ['slow'], 'fr': ['lent'], 'es': ['lento'], 'de': ['langsam'],
    'zh': ['慢'], 'ja': ['遅い'], 'ko': ['느리다'],
}

dicts = [(adj_dict, ans_dict), (adj_dict2, ans_dict2), (adj_dict3, ans_dict3),
         (adj_dict4, ans_dict4), (adj_dict5, ans_dict5), (adj_dict6, ans_dict6),
         (adj_dict7, ans_dict7), (adj_dict8, ans_dict8), (adj_dict9, ans_dict9),
         (adj_dict10, ans_dict10), (adj_dict11, ans_dict11), (adj_dict12, ans_dict12),]

adj_dict = {
    'en': 'small', 'fr': 'petit', 'es': 'pequeño', 'de': 'klein', 
    'zh': '小', 'ja': '小さい', 'ko': '작다',
}
ans_dict = {
    'en': ['big', 'large'], 'fr': ['grand', 'gros'], 'es': ['grande', 'enorme', 'gigante'], 
    'de': ['groß'], 'zh': ['大'], 'ja': ['大きい'], 'ko': ['크다', '큰']
}

adj_dict2 = {
    'en': 'heavy', 'fr': 'lourd', 'es': 'pesado', 'de': 'schwer', 
    'zh': '重', 'ja': '重い', 'ko': '무거운',
} # simple is the translation of einfach
ans_dict2 = {
    'en': ['light', 'simple'], 'fr': ['léger'], 'es': ['ligero'], 
    'de': ['leicht', 'einfach'], 'zh': ['轻', '輕'], 'ja': ['軽い'], 'ko': ['가벼운']
}

adj_dict3 = {
    'en': 'hot', 'fr': 'chaud', 'es': 'caliente', 'de': 'heiß', 
    'zh': '热', 'ja': '熱い', 'ko': '뜨거운',
}
ans_dict3 = {
    'en': ['cold', 'cool'], 'fr': ['froid'], 'es': ['frío', 'helado', 'fresco'], 'de': ['kalt', 'kühl'],
    'zh': ['冷', '凉', '寒'], 'ja': ['冷たい', '寒い'], 'ko': ['차가운'], 
}

adj_dict4 = {
    'en': 'new', 'fr': 'nouveau', 'es': 'nuevo', 'de': 'neu',
    'zh': '新', 'ja': '新しい', 'ko': '새롭다',
}
ans_dict4 = {
    'en': ['old', 'used'], 'fr': ['vieux', 'ancien'], 'es': ['viejo', 'antiguo'], 'de': ['alt'],
    'zh': ['旧', '老', '古'], 'ja': ['古い'], 'ko': ['오래되다', '낡다'],
}

adj_dict5 = {
    'en': 'soft', 'fr': 'doux', 'es': 'suave', 'de': 'weich',
    'zh': '软', 'ja': '柔らかい', 'ko': '부드럽다',
}
ans_dict5 = {
    'en': ['hard', 'firm'], 'fr': ['dur', 'amer', 'acide', 'aigre', 'fort', 'violent', 'brutal', 'raide'], 'es': ['duro', 'fuerte', 'áspero', 'rugoso', 'brusco', 'violento', 'agrio'], 'de': ['hart'],
    'zh': ['硬', '坚', '强'], 'ja': ['硬い', '固い', '堅い'], 'ko': ['딱딱하다', '단단하다'],
}
# 'nah' being understood as English
adj_dict6 = {
    'en': 'near', 'fr': 'près', 'es': 'cerca', 'de': 'nah', 
    'zh': '近', 'ja': '近い', 'ko': '가깝다'
}
ans_dict6 = {
    'en': ['far'], 'fr': ['loin', 'éloigné'], 'es': ['lejos'], 'de': ['weit', 'fern'], 
    'zh': ['远', '遥'], 'ja': ['遠い'], 'ko': ['멀다', '먼'],
}

adj_dict7 = {
    'en': 'dark', 'fr': 'foncé', 'es': 'oscuro', 'de': 'dunkel',
    'zh': '暗', 'ja': '暗い', 'ko': '어둡다',
}
ans_dict7 = {
    'en': ['light', 'bright'], 'fr': ['clair', 'pâle', 'brillant'], 'es': ['claro', 'luminoso', 'brillante'],
    'de': ['hell'], 'zh': ['亮', '明', '公开'], 'ja': ['明るい'], 'ko': ['밝다'],
}

adj_dict8 = {
    'en': 'young', 'fr': 'jeune', 'es': 'joven', 'de': 'jung',
    'zh': '年轻', 'ja': '若い', 'ko': '젊다',
}
ans_dict8 = {
    'en': ['old', 'mature'], 'fr': ['vieux', 'vielle', 'âgé', 'ancien'], 'es': ['viejo', 'mayor'], 'de': ['alt'],
    'zh': ['老', '老年', '年长'], 'ja': ['年を取った', '高齢の', '老いた'], 'ko': ['늙다', '오래되다'], 
}

# French produced wrong answer
adj_dict9 = {
    'en': 'quiet', 'fr': 'silencieux', 'es': 'tranquilo', 'de': 'leise',
    'zh': '安静', 'ja': '静かな', 'ko': '조용하다',
}
ans_dict9 = {
    'en': ['loud', 'noisy'], 'fr': ['bruyant', 'fort', 'tapageur'], 'es': ['ruidoso', 'nervioso', 'agitado', 'inquieto'], 'de': ['laut'],
    'zh': ['吵闹', '大声', '嘈杂', '热闹', '喧闹'], 'ja': ['うるさい', '騒がしい', '賑やかな'], 'ko': ['시끄럽다', '떠들썩하다', '활기차다'],
}

adj_dict10 = {
    'en': 'true', 'fr': 'vrai', 'es': 'verdadero', 'de': 'wahr',
    'zh': '真', 'ja': '本当の', 'ko': '진실하다',
}
ans_dict10 = {
    'en': ['false'], 'fr': ['faux'], 'es': ['falso'], 'de': ['falsch'],
    'zh': ['假', '虚'], 'ja': ['偽の', '嘘の', '間違った', '虚'], 'ko': ['거짓되다', '가짜이다', '불성실하다'],
}
# Japanese failed
adj_dict11 = {
    'en': 'active', 'fr': 'actif', 'es': 'activo', 'de': 'aktiv',
    'zh': '主动', 'ja': '能動的な', 'ko': '능동적이다',
}
ans_dict11 = {
    'en': ['passive', 'inactive', 'dormant'], 'fr': ['passif', 'inactif', 'sédentaire', 'paresseux'], 'es': ['pasivo', 'inactivo', 'sedentario', 'ocioso', 'latente'], 'de': ['passiv', 'inaktiv'],
    'zh': ['被动'], 'ja': ['受動的な', '消極的な', '怠惰な'], 'ko': ['수동적이다', '소극적이다'],
}

adj_dict12= {
    'en': 'fast', 'fr': 'rapide', 'es': 'rápido', 'de': 'schnell',
    'zh': '快', 'ja': '速い', 'ko': '빠르다',
}
ans_dict12 = {
    'en': ['slow'], 'fr': ['lent'], 'es': ['lento'], 'de': ['langsam'],
    'zh': ['慢'], 'ja': ['遅い'], 'ko': ['느리다'],
}
# 'gut' being understood as English
good_adj_dict = {
    'en': 'good', 'fr': 'bon', 'es': 'bueno', 'de': 'gut',
    'zh': '好', 'ja': '良い', 'ko': '선'
}
bad_ans_dict = {
    'en': ['bad'], 'fr': ['mauvais', 'mal'], 'es': ['malo'], 'de': ['schlecht'],
    'zh': ['不好', '坏'], 'ja': ['悪い'], 'ko': ['악']
}

# 2. Up / Down (Directional/Positional)
up_adj_dict = {
    'en': 'up', 'fr': 'haut', 'es': 'alto', 'de': 'hoch',
    'zh': '上', 'ja': '上に', 'ko': '상'
}
down_ans_dict = {
    'en': ['down', 'below'], 'fr': ['bas'], 'es': ['bajo'], 'de': ['tief', 'niedrig'],
    'zh': ['下'], 'ja': ['下に'], 'ko': ['하']
}

# 4. Rich / Poor
rich_adj_dict = {
    'en': 'rich', 'fr': 'riche', 'es': 'rico', 'de': 'reich',
    'zh': '富', 'ja': '豊か', 'ko': '부'
}
poor_ans_dict = {
    'en': ['poor'], 'fr': ['pauvre'], 'es': ['pobre'], 'de': ['arm'],
    'zh': ['贫'], 'ja': ['貧しい'], 'ko': ['빈']
}
# 'tot' being understood as English
# 5. Dead / Alive
dead_adj_dict = {
    'en': 'dead', 'fr': 'mort', 'es': 'muerto', 'de': 'tot',
    'zh': '死', 'ja': '死ぬ', 'ko': '사'
}
alive_ans_dict = {
    'en': ['alive'], 'fr': ['vivant'], 'es': ['vivo'], 'de': ['lebendig'],
    'zh': ['生'], 'ja': ['生きる', '生'], 'ko': ['생']
}

# French, German produces wrong output
# 1. Whole / Part (Completeness)
whole_adj_dict = {
    'en': 'whole', 'fr': 'entier', 'es': 'entero', 'de': 'ganz',
    'zh': '全体', 'ja': '全体', 'ko': '전' # '全/全/전' as in 全部/全体/전체 (whole/entire)
}
part_ans_dict = {
    'en': ['part'], 'fr': 'partiel', 'es': 'parcial', 'de': 'teil', # 'teil' is a noun/prefix, 'teilweise' is adj
    'zh': '部分', 'ja': '部分', 'ko': '부' # '部/部/부' as in 部分/部分/부분 (part)
}

# fr and de interpret as hard (as in texture)
# 9. Solid / Liquid (State of Matter)
solid_adj_dict = {
    'en': 'solid', 'fr': 'solide', 'es': 'sólido', 'de': 'fest',
    'zh': '固体', 'ja': '固体', 'ko': '고' # '固/固/고' as in 固体/固体/고체 (solid)
}
liquid_ans_dict = {
    'en': 'liquid', 'fr': ['liquide', 'fragile'], 'es': 'líquido', 'de': ['flüssig', 'weich'],
    'zh': '液体', 'ja': '液体', 'ko': '액' # '液/液/액' as in 液体/液体/액체 (liquid)
}

# 10. Empty / Full (Container)
empty_adj_dict = {
    'en': 'empty', 'fr': 'vide', 'es': 'vacío', 'de': 'leer',
    'zh': '空', 'ja': '空っぽ', 'ko': '공'
}
full_ans_dict = {
    'en': ['full'], 'fr': ['plein'], 'es': ['lleno'], 'de': ['voll'],
    'zh': ['满'], 'ja': ['満杯', 'いっぱい'], 'ko': ['만']
}

# English returns 'gay' as the first prediction
# 6. Straight / Curved (Line/Path)
straight_line_adj_dict = {
    'en': 'straight', 'fr': 'droit', 'es': 'recto', 'de': 'gerade',
    'zh': '直', 'ja': '直線', 'ko': '직'
}
curved_line_ans_dict = {
    'en': 'curved', 'fr': 'courbe', 'es': 'curvo', 'de': 'gekrümmt',
    'zh': '弯', 'ja': '曲線', 'ko': '곡'
}

# 2. Male / Female (Biological Gender)
male_adj_dict = {
    'en': 'male', 'fr': 'mâle', 'es': 'macho', 'de': 'männlich',
    'zh': '雄', 'ja': '男性', 'ko': '웅' # '雄/雄/웅' as in 雄性/雄性/웅성 (male)
}
female_ans_dict = {
    'en': ['female', 'feminine'], 'fr': ['femelle'], 'es': ['hembra'], 'de': ['weiblich'],
    'zh': ['雌'], 'ja': ['女性'], 'ko': ['자'] # '雌/雌/자' as in 雌性/雌性/자성 (female)
}

# English, French, German predicts the same
# 9. Formal / Informal (Style)
formal_adj_dict = {
    'en': 'formal', 'fr': 'formel', 'es': 'formal', 'de': 'formell',
    'zh': '正式', 'ja': '正式', 'ko': '정' # '正' as in 正式 (formal)
}
informal_ans_dict = {
    'en': 'informal', 'fr': 'informel', 'es': 'informal', 'de': 'informell',
    'zh': '非正式', 'ja': '非正式', 'ko': '비' # '非' as in 非正式 (informal)
}
# not being understood
wet_adj_dict = {
    'en': 'wet', 'fr': 'mouillé', 'es': 'mojado', 'de': 'nass',
    'zh': '湿', 'ja': '湿った', 'ko': '습'
}
dry_ans_dict = {
    'en': ['dry'], 'fr': ['sec'], 'es': ['seco'], 'de': ['trocken'],
    'zh': ['干', '干燥'], 'ja': ['乾いた', '乾燥'], 'ko': ['건']
}

# 2. Open / Closed (State)
open_adj_dict = {
    'en': 'open', 'fr': 'ouvert', 'es': 'abierto', 'de': 'offen',
    'zh': '开', 'ja': '開く', 'ko': '개'
}
closed_ans_dict = {
    'en': ['closed', 'close'], 'fr': ['fermé'], 'es': ['cerrado'], 'de': ['geschlossen'],
    'zh': ['关', '闭'], 'ja': ['閉じる', '閉'], 'ko': ['폐']
}

awake_adj_dict = {
    'en': 'awake', 'fr': 'éveillé', 'es': 'despierto', 'de': 'wach',
    'zh': '清醒', 'ja': '覚醒', 'ko': '각' # '醒/覚/각' as in 清醒/覚醒/각성 (awake/conscious)
}
asleep_ans_dict = {
    'en': ['asleep', 'sleepy', 'sleep', 'unconscious'], 'fr': ['endormi'], 'es': ['dormido'], 'de': ['schlafend'],
    'zh': ['昏迷'], 'ja': ['眠', '鎮'], 'ko': ['수'] # '眠/眠/수' as in 睡眠/睡眠/수면 (sleep)
}

# german makes a mistake
tight_adj_dict = {
    'en': 'tight', 'fr': 'serré', 'es': 'apretado', 'de': 'eng',
    'zh': '紧密', 'ja': '締まる', 'ko': '긴' # '紧/締/긴' as in 紧密/締まる/긴장 (tight/tense)
}
loose_ans_dict = {
    'en': ['loose'], 'fr': ['lâche', 'détendu'], 'es': ['suelto'], 'de': ['locker'],
    'zh': ['宽松'], 'ja': ['緩む'], 'ko': ['이'] # '松/緩/이' as in 宽松/緩む/이완 (loose/relaxed)
}

# 2. Healthy / Sick (Health State)
healthy_adj_dict = {
    'en': 'healthy', 'fr': 'sain', 'es': 'sano', 'de': 'gesund',
    'zh': '健康', 'ja': '健康', 'ko': '건' # '康/健/건' as in 健康/健康/건강 (healthy)
}
sick_ans_dict = {
    'en': ['sick', 'unhealthy', 'ill'], 'fr': ['malade', 'sale', 'malsain'], 'es': ['enfermo'], 'de': ['krank'],
    'zh': ['疾病', '病', '不健康'], 'ja': ['病弱', '病気', '不健康'], 'ko': ['병']
}

# 10. Positive / Negative (Result/Polarity)
positive_adj_dict = {
    'en': 'positive', 'fr': 'positif', 'es': 'positivo', 'de': 'positiv',
    'zh': '正', 'ja': '正', 'ko': '정'
}
negative_ans_dict = {
    'en': ['negative'], 'fr': ['négatif'], 'es': ['negativo'], 'de': ['negativ'],
    'zh': ['负', '反', '邪', '不正'], 'ja': ['負', '反', '逆', '邪'], 'ko': ['부']
}

# tokens for English, French, German are same
# 7. Optimistic / Pessimistic (Outlook)
optimistic_adj_dict = {
    'en': 'optimistic', 'fr': 'optimiste', 'es': 'optimista', 'de': 'optimistisch',
    'zh': '乐观', 'ja': '楽観的', 'ko': '낙' # '乐/楽/낙' as in 乐观/楽観/낙관 (optimistic)
}
pessimistic_ans_dict = {
    'en': 'pessimistic', 'fr': 'pessimiste', 'es': 'pesimista', 'de': 'pessimistisch',
    'zh': '悲观', 'ja': '悲観的', 'ko': '비' # '悲/悲/비' as in 悲观/悲観/비관 (pessimistic)
}

# English and French are the same + Japanese does not work
# 6. Responsible / Irresponsible (Accountability)
responsible_adj_dict = {
    'en': 'responsible', 'fr': 'responsable', 'es': 'responsable', 'de': 'verantwortlich',
    'zh': '责任', 'ja': '責任', 'ko': '책' # '责/責/책' as in 责任/責任/책임 (responsibility)
}
irresponsible_ans_dict = {
    'en': 'irresponsible', 'fr': 'irresponsable', 'es': 'irresponsable', 'de': 'unverantwortlich',
    'zh': '不负责', 'ja': '無責任', 'ko': '무' # '不/無/무' as in 不负责/無責任/무책임 (irresponsible)
}

# French does not work
# 3. Natural / Artificial (Origin)
natural_adj_dict = {
    'en': 'natural', 'fr': 'naturel', 'es': 'natural', 'de': 'natürlich',
    'zh': '天然', 'ja': '天然', 'ko': '천' # '天' as in 天然 (natural)
}
artificial_ans_dict = {
    'en': 'artificial', 'fr': 'artificiel', 'es': 'artificial', 'de': 'künstlich',
    'zh': '人工', 'ja': '人工', 'ko': '인' # '人' as in 人工 (artificial)
}

# English and French are same token
# 2. Simple / Complex (Complexity)
simple_adj_dict = {
    'en': 'simple', 'fr': 'simple', 'es': 'simple', 'de': 'einfach',
    'zh': '简', 'ja': '単純', 'ko': '간'
}
complex_ans_dict = {
    'en': 'complex', 'fr': 'complexe', 'es': 'complejo', 'de': 'komplex',
    'zh': '繁', 'ja': '複雑', 'ko': '복'
}

# 10. Top / Bottom (Position)
top_adj_dict = {
    'en': 'top', 'fr': 'haut', 'es': 'superior', 'de': 'oben',
    'zh': '顶', 'ja': '頂点', 'ko': '정'
}
bottom_ans_dict = {
    'en': ['bottom', 'below'], 'fr': ['bas'], 'es': ['inferior'], 'de': ['unten'],
    'zh': ['底'], 'ja': ['底辺'], 'ko': ['저']
}

# French does not work
# 9. Rough / Smooth (Texture)
rough_adj_dict = {
    'en': 'rough', 'fr': 'rugueux', 'es': 'áspero', 'de': 'rau',
    'zh': '粗', 'ja': '粗い', 'ko': '조'
}
smooth_ans_dict = {
    'en': 'smooth', 'fr': 'lisse', 'es': 'liso', 'de': 'glatt',
    'zh': '光', 'ja': '滑らか', 'ko': '활'
}

# 3. Front / Back (Position)
front_adj_dict = {
    'en': 'front', 'fr': 'avant', 'es': 'frontal', 'de': 'vorn',
    'zh': '前', 'ja': '前方', 'ko': '전'
}
back_ans_dict = {
    'en': ['back', 'rear'], 'fr': ['arrière', 'après', 'derrière'], 'es': ['trasero'], 'de': ['hinten', 'hinter'],
    'zh': ['后'], 'ja': ['後方'], 'ko': ['후']
}

# 1. High / Low (Elevation/Rank)
high_adj_dict = {
    'en': 'high', 'fr': 'haut', 'es': 'alto', 'de': 'hoch',
    'zh': '高', 'ja': '高い', 'ko': '고'
}
low_ans_dict = {
    'en': ['low'], 'fr': ['bas'], 'es': ['bajo'], 'de': ['tief', 'niedrig'],
    'zh': ['低'], 'ja': ['低い', '安い'], 'ko': ['저']
}

# 8. Early / Late
early_adj_dict = {
    'en': 'early', 'fr': 'précoce', 'es': 'temprano', 'de': 'früh',
    'zh': '早', 'ja': '早い', 'ko': '조'
}
late_ans_dict = {
    'en': ['late', 'slow'], 'fr': ['tardif'], 'es': ['tarde'], 'de': ['spät'],
    'zh': ['晚'], 'ja': ['遅い'], 'ko': ['만']
}

# German does not work
# 7. Beautiful / Ugly
beautiful_adj_dict = {
    'en': 'beautiful', 'fr': 'beau', 'es': 'bello', 'de': 'schön',
    'zh': '美', 'ja': '美しい', 'ko': '미'
}
ugly_ans_dict = {
    'en': ['ugly'], 'fr': ['laid', 'mauvais'], 'es': ['feo'], 'de': ['hässlich'],
    'zh': ['丑'], 'ja': ['醜い'], 'ko': ['추']
}

# 3. Easy / Difficult
easy_adj_dict = {
    'en': 'easy', 'fr': 'facile', 'es': 'fácil', 'de': 'einfach',
    'zh': '易', 'ja': '簡単', 'ko': '이'
}
difficult_ans_dict = {
    'en': ['difficult'], 'fr': ['difficile'], 'es': ['difícil'], 'de': ['schwer'],
    'zh': ['难'], 'ja': ['難解', '難しい'], 'ko': ['난']
}

small_data = [(adj_dict, ans_dict), (adj_dict2, ans_dict2), (adj_dict3, ans_dict3),
         (adj_dict4, ans_dict4), (adj_dict5, ans_dict5), (adj_dict6, ans_dict6),
         (adj_dict7, ans_dict7), (adj_dict8, ans_dict8), (adj_dict9, ans_dict9),
         (adj_dict10, ans_dict10), (adj_dict11, ans_dict11), (adj_dict12, ans_dict12),]

train_data = [(adj_dict, ans_dict), (adj_dict2, ans_dict2), (adj_dict3, ans_dict3), 
              (adj_dict4, ans_dict4), (adj_dict5, ans_dict5), (adj_dict6, ans_dict6), 
              (adj_dict7, ans_dict7), (adj_dict8, ans_dict8), (adj_dict10, ans_dict10), 
              (adj_dict12, ans_dict12), (good_adj_dict, bad_ans_dict), (up_adj_dict, down_ans_dict),
              (rich_adj_dict, poor_ans_dict), (dead_adj_dict, alive_ans_dict), (empty_adj_dict, full_ans_dict), 
              (male_adj_dict, female_ans_dict), (wet_adj_dict, dry_ans_dict), (open_adj_dict, closed_ans_dict), 
              (awake_adj_dict, asleep_ans_dict), (healthy_adj_dict, sick_ans_dict)]
test_data = [(positive_adj_dict, negative_ans_dict), (top_adj_dict, bottom_ans_dict), (front_adj_dict, back_ans_dict), 
             (high_adj_dict, low_ans_dict), (early_adj_dict, late_ans_dict), (easy_adj_dict, difficult_ans_dict)]


big_data_2 = [
    # --- Original 10 Pairs ---
    (
        {'en': 'good', 'fr': 'bon', 'de': 'gut', 'zh': '好', 'ja': '良い', 'es': 'bueno', 'ko': '좋은'},
        {'en': ['bad'], 'fr': ['mauvais'], 'de': ['schlecht'], 'zh': ['坏'], 'ja': ['悪い'], 'es': ['malo'], 'ko': ['나쁜']}
    ),
    (
        {'en': 'happy', 'fr': 'heureux', 'de': 'glücklich', 'zh': '开心', 'ja': '嬉しい', 'es': 'feliz', 'ko': '행복한'},
        {'en': ['sad', 'unhappy'], 'fr': ['triste', 'malheureux'], 'de': ['traurig', 'unglücklich'], 'zh': ['难过', '不高兴'], 'ja': ['悲しい'], 'es': ['triste', 'infeliz'], 'ko': ['슬픈', '불행한']}
    ),
    (
        {'en': 'big', 'fr': 'grand', 'de': 'groß', 'zh': '大', 'ja': '大きい', 'es': 'grande', 'ko': '큰'},
        {'en': ['small'], 'fr': ['petit'], 'de': ['klein'], 'zh': ['小'], 'ja': ['小さい'], 'es': ['pequeño'], 'ko': ['작은']}
    ),
    (
        {'en': 'hot', 'fr': 'chaud', 'de': 'heiß', 'zh': '热', 'ja': '暑い', 'es': 'caliente', 'ko': '더운'},
        {'en': ['cold'], 'fr': ['froid'], 'de': ['kalt'], 'zh': ['冷'], 'ja': ['寒い', '冷たい'], 'es': ['frío'], 'ko': ['추운', '차가운']}
    ),
    (
        {'en': 'fast', 'fr': 'rapide', 'de': 'schnell', 'zh': '快', 'ja': '速い', 'es': 'rápido', 'ko': '빠른'},
        {'en': ['slow'], 'fr': ['lent', 'lente'], 'de': ['langsam'], 'zh': ['慢'], 'ja': ['遅い'], 'es': ['lento'], 'ko': ['느린']}
    ),
    (
        {'en': 'light', 'fr': 'léger', 'de': 'leicht', 'zh': '轻', 'ja': '軽い', 'es': 'ligero', 'ko': '가벼운'},
        {'en': ['heavy', 'dark'], 'fr': ['lourd'], 'de': ['schwer'], 'zh': ['重'], 'ja': ['重い'], 'es': ['pesado'], 'ko': ['무거운']}
    ),
    (
        {'en': 'easy', 'fr': 'facile', 'de': 'einfach', 'zh': '容易', 'ja': '簡単な', 'es': 'fácil', 'ko': '쉬운'},
        {'en': ['difficult', 'hard'], 'fr': ['difficile'], 'de': ['schwierig', 'schwer'], 'zh': ['难'], 'ja': ['難しい'], 'es': ['difícil'], 'ko': ['어려운']}
    ),
    (
        {'en': 'new', 'fr': 'nouveau', 'de': 'neu', 'zh': '新', 'ja': '新しい', 'es': 'nuevo', 'ko': '새로운'},
        {'en': ['old'], 'fr': ['vieux', 'ancien'], 'de': ['alt'], 'zh': ['旧'], 'ja': ['古い'], 'es': ['viejo'], 'ko': ['오래된']}
    ),
    (
        {'en': 'true', 'fr': 'vrai', 'de': 'wahr', 'zh': '真', 'ja': '正しい', 'es': 'verdadero', 'ko': '진실한'},
        {'en': ['false', 'untrue'], 'fr': ['faux'], 'de': ['falsch', 'unwahr'], 'zh': ['假'], 'ja': ['間違った'], 'es': ['falso', 'incorrecto'], 'ko': ['거짓의', '틀린']}
    ),
    (
        {'en': 'alive', 'fr': 'vivant', 'de': 'lebendig', 'zh': '活', 'ja': '生きている', 'es': 'vivo', 'ko': '살아있는'},
        {'en': ['dead'], 'fr': ['mort'], 'de': ['tot'], 'zh': ['死'], 'ja': ['死んでいる'], 'es': ['muerto'], 'ko': ['죽은']}
    ),
    # --- Added 6 more original pairs ---
    (
        {'en': 'full', 'fr': 'plein', 'de': 'voll', 'zh': '满', 'ja': '満杯', 'es': 'lleno', 'ko': '가득 찬'},
        {'en': ['empty'], 'fr': ['vide'], 'de': ['leer'], 'zh': ['空'], 'ja': ['空っぽ'], 'es': ['vacío'], 'ko': ['빈']}
    ),
    (
        {'en': 'bright', 'fr': 'brillant', 'de': 'hell', 'zh': '亮', 'ja': '明るい', 'es': 'brillante', 'ko': '밝은'},
        {'en': ['dark', 'dim'], 'fr': ['sombre', 'obscur'], 'de': ['dunkel'], 'zh': ['暗'], 'ja': ['暗い'], 'es': ['oscuro', 'opaco'], 'ko': ['어두운', '흐릿한']}
    ),
    (
        {'en': 'strong', 'fr': 'fort', 'de': 'stark', 'zh': '强', 'ja': '強い', 'es': 'fuerte', 'ko': '강한'},
        {'en': ['weak'], 'fr': ['faible'], 'de': ['schwach'], 'zh': ['弱'], 'ja': ['弱い'], 'es': ['débil'], 'ko': ['약한']}
    ),
    (
        {'en': 'clean', 'fr': 'propre', 'de': 'sauber', 'zh': '干净', 'ja': 'きれいな', 'es': 'limpio', 'ko': '깨끗한'},
        {'en': ['dirty'], 'fr': ['sale'], 'de': ['schmutzig'], 'zh': ['脏'], 'ja': ['汚い'], 'es': ['sucio'], 'ko': ['더러운']}
    ),
    (
        {'en': 'open', 'fr': 'ouvert', 'de': 'offen', 'zh': '开', 'ja': '開いた', 'es': 'abierto', 'ko': '열린'},
        {'en': ['closed'], 'fr': ['fermé'], 'de': ['geschlossen'], 'zh': ['关'], 'ja': ['閉じた'], 'es': ['cerrado'], 'ko': ['닫힌']}
    ),
    (
        {'en': 'rich', 'fr': 'riche', 'de': 'reich', 'zh': '富裕', 'ja': '裕福な', 'es': 'rico', 'ko': '부유한'},
        {'en': ['poor'], 'fr': ['pauvre'], 'de': ['arm'], 'zh': ['贫穷'], 'ja': ['貧しい'], 'es': ['pobre'], 'ko': ['가난한']}
    ),
    # --- New original pairs to reach a base of ~25-30 ---
    (
        {'en': 'beautiful', 'fr': 'beau', 'de': 'schön', 'zh': '美', 'ja': '美しい', 'es': 'hermoso', 'ko': '아름다운'},
        {'en': ['ugly'], 'fr': ['laid'], 'de': ['hässlich'], 'zh': ['丑'], 'ja': ['醜い'], 'es': ['feo'], 'ko': ['못생긴']}
    ),
    (
        {'en': 'long', 'fr': 'long', 'de': 'lang', 'zh': '长', 'ja': '長い', 'es': 'largo', 'ko': '긴'},
        {'en': ['short'], 'fr': ['court'], 'de': ['kurz'], 'zh': ['短'], 'ja': ['短い'], 'es': ['corto'], 'ko': ['짧은']}
    ),
    (
        {'en': 'wide', 'fr': 'large', 'de': 'breit', 'zh': '宽', 'ja': '広い', 'es': 'ancho', 'ko': '넓은'},
        {'en': ['narrow'], 'fr': ['étroit'], 'de': ['eng'], 'zh': ['窄'], 'ja': ['狭い'], 'es': ['estrecho'], 'ko': ['좁은']}
    ),
    (
        {'en': 'hard', 'fr': 'dur', 'de': 'hart', 'zh': '硬', 'ja': '硬い', 'es': 'duro', 'ko': '딱딱한'},
        {'en': ['soft'], 'fr': ['mou'], 'de': ['weich'], 'zh': ['软'], 'ja': ['柔らかい'], 'es': ['suave'], 'ko': ['부드러운']}
    ),
    (
        {'en': 'dry', 'fr': 'sec', 'de': 'trocken', 'zh': '干', 'ja': '乾いた', 'es': 'seco', 'ko': '마른'},
        {'en': ['wet', 'moist'], 'fr': ['humide', 'mouillé'], 'de': ['nass'], 'zh': ['湿'], 'ja': ['濡れた', '湿った'], 'es': ['mojado', 'húmedo'], 'ko': ['젖은', '축축한']}
    ),
    (
        {'en': 'loud', 'fr': 'fort', 'de': 'laut', 'zh': '大声', 'ja': 'うるさい', 'es': 'ruidoso', 'ko': '시끄러운'},
        {'en': ['quiet', 'silent'], 'fr': ['silencieux', 'doux'], 'de': ['leise', 'still'], 'zh': ['安静', '小声'], 'ja': ['静かな'], 'es': ['tranquilo', 'silencioso'], 'ko': ['조용한', '고요한']}
    ),
    (
        {'en': 'strong', 'fr': 'fort', 'de': 'stark', 'zh': '强壮', 'ja': '丈夫な', 'es': 'fuerte', 'ko': '튼튼한'},
        {'en': ['weak'], 'fr': ['faible'], 'de': ['schwach'], 'zh': ['虚弱'], 'ja': ['弱い'], 'es': ['débil'], 'ko': ['약한']}
    ),
    (
        {'en': 'early', 'fr': 'tôt', 'de': 'früh', 'zh': '早', 'ja': '早い', 'es': 'temprano', 'ko': '이른'},
        {'en': ['late'], 'fr': ['tard'], 'de': ['spät'], 'zh': ['晚'], 'ja': ['遅い'], 'es': ['tarde'], 'ko': ['늦은']}
    ),
    (
        {'en': 'near', 'fr': 'proche', 'de': 'nah', 'zh': '近', 'ja': '近い', 'es': 'cerca', 'ko': '가까운'},
        {'en': ['far'], 'fr': ['loin'], 'de': ['fern', 'weit'], 'zh': ['远'], 'ja': ['遠い'], 'es': ['lejos'], 'ko': ['먼']}
    ),
    (
        {'en': 'deep', 'fr': 'profond', 'de': 'tief', 'zh': '深', 'ja': '深い', 'es': 'profundo', 'ko': '깊은'},
        {'en': ['shallow'], 'fr': ['peu profond'], 'de': ['flach'], 'zh': ['浅'], 'ja': ['浅い'], 'es': ['superficial', 'poco profundo'], 'ko': ['얕은']}
    ),

    # --- Flipping existing pairs & adding new ones to reach 50 ---

    # 1. (Flipped) Bad / Good
    (
        {'en': 'bad', 'fr': 'mauvais', 'de': 'schlecht', 'zh': '坏', 'ja': '悪い', 'es': 'malo', 'ko': '나쁜'},
        {'en': ['good'], 'fr': ['bon'], 'de': ['gut'], 'zh': ['好'], 'ja': ['良い'], 'es': ['bueno'], 'ko': ['좋은']}
    ),
    # 2. (Flipped) Sad / Happy
    (
        {'en': 'sad', 'fr': 'triste', 'de': 'traurig', 'zh': '难过', 'ja': '悲しい', 'es': 'triste', 'ko': '슬픈'},
        {'en': ['happy', 'joyful'], 'fr': ['heureux', 'joyeux'], 'de': ['glücklich'], 'zh': ['开心'], 'ja': ['嬉しい'], 'es': ['feliz', 'alegre'], 'ko': ['행복한', '기쁜']}
    ),
    # 3. (Flipped) Small / Big
    (
        {'en': 'small', 'fr': 'petit', 'de': 'klein', 'zh': '小', 'ja': '小さい', 'es': 'pequeño', 'ko': '작은'},
        {'en': ['big', 'large'], 'fr': ['grand'], 'de': ['groß'], 'zh': ['大'], 'ja': ['大きい'], 'es': ['grande'], 'ko': ['큰']}
    ),
    # 4. (Flipped) Cold / Hot
    (
        {'en': 'cold', 'fr': 'froid', 'de': 'kalt', 'zh': '冷', 'ja': '寒い', 'es': 'frío', 'ko': '추운'},
        {'en': ['hot', 'warm'], 'fr': ['chaud'], 'de': ['heiß'], 'zh': ['热'], 'ja': ['暑い', '熱い'], 'es': ['caliente', 'cálido'], 'ko': ['더운', '따뜻한']}
    ),
    # 5. (Flipped) Slow / Fast
    (
        {'en': 'slow', 'fr': 'lent', 'de': 'langsam', 'zh': '慢', 'ja': '遅い', 'es': 'lento', 'ko': '느린'},
        {'en': ['fast', 'quick'], 'fr': ['rapide'], 'de': ['schnell'], 'zh': ['快'], 'ja': ['速い', '早い'], 'es': ['rápido', 'veloz'], 'ko': ['빠른']}
    ),
    # 6. (Flipped) Heavy / Light
    (
        {'en': 'heavy', 'fr': 'lourd', 'de': 'schwer', 'zh': '重', 'ja': '重い', 'es': 'pesado', 'ko': '무거운'},
        {'en': ['light'], 'fr': ['léger'], 'de': ['leicht'], 'zh': ['轻'], 'ja': ['軽い'], 'es': ['ligero'], 'ko': ['가벼운']}
    ),
    # 7. (Flipped) Difficult / Easy
    (
        {'en': 'difficult', 'fr': 'difficile', 'de': 'schwierig', 'zh': '难', 'ja': '難しい', 'es': 'difícil', 'ko': '어려운'},
        {'en': ['easy', 'simple'], 'fr': ['facile'], 'de': ['einfach'], 'zh': ['容易', '易'], 'ja': ['簡単な', '簡単'], 'es': ['fácil', 'sencillo'], 'ko': ['쉬운', '간단한']}
    ),
    # 8. (Flipped) Old / New
    (
        {'en': 'old', 'fr': 'vieux', 'de': 'alt', 'zh': '旧', 'ja': '古い', 'es': 'viejo', 'ko': '오래된'},
        {'en': ['new'], 'fr': ['nouveau', 'jeune'], 'de': ['neu'], 'zh': ['新'], 'ja': ['新しい'], 'es': ['nuevo', 'joven'], 'ko': ['새로운', '젊은']}
    ),
    # 9. (Flipped) False / True
    (
        {'en': 'false', 'fr': 'faux', 'de': 'falsch', 'zh': '假', 'ja': '間違った', 'es': 'falso', 'ko': '거짓의'},
        {'en': ['true', 'correct'], 'fr': ['vrai', 'correct'], 'de': ['wahr', 'richtig'], 'zh': ['真'], 'ja': ['正しい'], 'es': ['verdadero', 'correcto'], 'ko': ['진실한', '올바른']}
    ),
    # 10. (Flipped) Dead / Alive
    (
        {'en': 'dead', 'fr': 'mort', 'de': 'tot', 'zh': '死', 'ja': '死んでいる', 'es': 'muerto', 'ko': '죽은'},
        {'en': ['alive', 'living'], 'fr': ['vivant'], 'de': ['lebendig'], 'zh': ['活', '生'], 'ja': ['生きている'], 'es': ['vivo', 'viviente'], 'ko': ['살아있는', '생생한']}
    ),
    # 11. (Flipped) Empty / Full
    (
        {'en': 'empty', 'fr': 'vide', 'de': 'leer', 'zh': '空', 'ja': '空っぽ', 'es': 'vacío', 'ko': '빈'},
        {'en': ['full'], 'fr': ['plein'], 'de': ['voll'], 'zh': ['满'], 'ja': ['満杯'], 'es': ['lleno'], 'ko': ['가득 찬']}
    ),
    # 12. (Flipped) Dark / Bright
    (
        {'en': 'dark', 'fr': 'sombre', 'de': 'dunkel', 'zh': '暗', 'ja': '暗い', 'es': 'oscuro', 'ko': '어두운'},
        {'en': ['bright', 'light'], 'fr': ['brillant', 'clair'], 'de': ['hell'], 'zh': ['亮', '光明'], 'ja': ['明るい'], 'es': ['brillante', 'claro'], 'ko': ['밝은']}
    ),
    # 13. (Flipped) Weak / Strong
    (
        {'en': 'weak', 'fr': 'faible', 'de': 'schwach', 'zh': '弱', 'ja': '弱い', 'es': 'débil', 'ko': '약한'},
        {'en': ['strong'], 'fr': ['fort'], 'de': ['stark'], 'zh': ['强', '强壮'], 'ja': ['強い', '丈夫な'], 'es': ['fuerte'], 'ko': ['강한']}
    ),
    # 14. (Flipped) Dirty / Clean
    (
        {'en': 'dirty', 'fr': 'sale', 'de': 'schmutzig', 'zh': '脏', 'ja': '汚い', 'es': 'sucio', 'ko': '더러운'},
        {'en': ['clean'], 'fr': ['propre'], 'de': ['sauber'], 'zh': ['干净'], 'ja': ['きれいな', 'きれい', '綺麗'], 'es': ['limpio'], 'ko': ['깨끗한']}
    ),
    # 15. (Flipped) Closed / Open
    (
        {'en': 'closed', 'fr': 'fermé', 'de': 'geschlossen', 'zh': '关', 'ja': '閉じた', 'es': 'cerrado', 'ko': '닫힌'},
        {'en': ['open'], 'fr': ['ouvert'], 'de': ['offen'], 'zh': ['开'], 'ja': ['開いた'], 'es': ['abierto'], 'ko': ['열린']}
    ),
    # 16. (Flipped) Poor / Rich
    (
        {'en': 'poor', 'fr': 'pauvre', 'de': 'arm', 'zh': '贫穷', 'ja': '貧しい', 'es': 'pobre', 'ko': '가난한'},
        {'en': ['rich', 'wealthy'], 'fr': ['riche'], 'de': ['reich'], 'zh': ['富裕'], 'ja': ['裕福な', '豊かな'], 'es': ['rico', 'adinerado'], 'ko': ['부유한', '풍부한']}
    ),
    # 17. (Flipped) Ugly / Beautiful
    (
        {'en': 'ugly', 'fr': 'laid', 'de': 'hässlich', 'zh': '丑', 'ja': '醜い', 'es': 'feo', 'ko': '못생긴'},
        {'en': ['beautiful', 'pretty'], 'fr': ['beau', 'joli'], 'de': ['schön'], 'zh': ['美'], 'ja': ['美しい'], 'es': ['hermoso', 'bonito'], 'ko': ['아름다운', '예쁜']}
    ),
    # 18. (Flipped) Short / Long
    (
        {'en': 'short', 'fr': 'court', 'de': 'kurz', 'zh': '短', 'ja': '短い', 'es': 'corto', 'ko': '짧은'},
        {'en': ['long'], 'fr': ['long'], 'de': ['lang'], 'zh': ['长'], 'ja': ['長い'], 'es': ['largo'], 'ko': ['긴']}
    ),
    # 19. (Flipped) Narrow / Wide
    (
        {'en': 'narrow', 'fr': 'étroit', 'de': 'eng', 'zh': '窄', 'ja': '狭い', 'es': 'estrecho', 'ko': '좁은'},
        {'en': ['wide', 'broad'], 'fr': ['large'], 'de': ['breit'], 'zh': ['宽'], 'ja': ['広い'], 'es': ['ancho'], 'ko': ['넓은']}
    ),
    # 20. (Flipped) Soft / Hard
    (
        {'en': 'soft', 'fr': 'mou', 'de': 'weich', 'zh': '软', 'ja': '柔らかい', 'es': 'suave', 'ko': '부드러운'},
        {'en': ['hard', 'firm'], 'fr': ['dur'], 'de': ['hart'], 'zh': ['硬'], 'ja': ['硬い'], 'es': ['duro'], 'ko': ['딱딱한']}
    ),
    # 21. (Flipped) Wet / Dry
    (
        {'en': 'wet', 'fr': 'humide', 'de': 'nass', 'zh': '湿', 'ja': '濡れた', 'es': 'mojado', 'ko': '젖은'},
        {'en': ['dry'], 'fr': ['sec'], 'de': ['trocken'], 'zh': ['干'], 'ja': ['乾いた'], 'es': ['seco'], 'ko': ['마른']}
    ),
    # 22. (Flipped) Quiet / Loud
    (
        {'en': 'quiet', 'fr': 'silencieux', 'de': 'leise', 'zh': '安静', 'ja': '静かな', 'es': 'tranquilo', 'ko': '조용한'},
        {'en': ['loud', 'noisy'], 'fr': ['fort', 'bruyant'], 'de': ['laut'], 'zh': ['大声', '吵闹'], 'ja': ['うるさい'], 'es': ['ruidoso'], 'ko': ['시끄러운']}
    ),
    # 23. (Flipped) Late / Early
    (
        {'en': 'late', 'fr': 'tard', 'de': 'spät', 'zh': '晚', 'ja': '遅い', 'es': 'tarde', 'ko': '늦은'},
        {'en': ['early'], 'fr': ['tôt'], 'de': ['früh'], 'zh': ['早'], 'ja': ['早い'], 'es': ['temprano'], 'ko': ['이른']}
    ),
    # 24. (Flipped) Far / Near
    (
        {'en': 'far', 'fr': 'loin', 'de': 'fern', 'zh': '远', 'ja': '遠い', 'es': 'lejos', 'ko': '먼'},
        {'en': ['near', 'close'], 'fr': ['proche'], 'de': ['nah'], 'zh': ['近'], 'ja': ['近い'], 'es': ['cerca'], 'ko': ['가까운']}
    ),
    # 25. (Flipped) Shallow / Deep
    (
        {'en': 'shallow', 'fr': 'peu profond', 'de': 'flach', 'zh': '浅', 'ja': '浅い', 'es': 'superficial', 'ko': '얕은'},
        {'en': ['deep'], 'fr': ['profond'], 'de': ['tief', 'hoch'], 'zh': ['深'], 'ja': ['深い'], 'es': ['profundo'], 'ko': ['깊은']}
    ),
    # --- New original pairs to fill out the remaining slots (25 more) ---
    (
        {'en': 'young', 'fr': 'jeune', 'de': 'jung', 'zh': '年轻', 'ja': '若い', 'es': 'joven', 'ko': '젊은'},
        {'en': ['old'], 'fr': ['vieux'], 'de': ['alt'], 'zh': ['老'], 'ja': ['老いた'], 'es': ['viejo'], 'ko': ['늙은']}
    ),
    (
        {'en': 'clean', 'fr': 'propre', 'de': 'sauber', 'zh': '干净', 'ja': '清潔な', 'es': 'limpio', 'ko': '청결한'},
        {'en': ['unclean'], 'fr': ['impropre'], 'de': ['unsauber'], 'zh': ['不干净'], 'ja': ['不潔な'], 'es': ['impuro', 'sucio'], 'ko': ['불결한', '더러운']}
    ),
    (
        {'en': 'kind', 'fr': 'gentil', 'de': 'freundlich', 'zh': '善良', 'ja': '親切な', 'es': 'amable', 'ko': '친절한'},
        {'en': ['unkind', 'mean'], 'fr': ['méchant'], 'de': ['unfreundlich'], 'zh': ['不善良', '邪恶'], 'ja': ['意地悪な', '不親切な', '失礼な'], 'es': ['desagradable', 'malo'], 'ko': ['불친절한', '못된']}
    ),
    (
        {'en': 'brave', 'fr': 'courageux', 'de': 'mutig', 'zh': '勇敢', 'ja': '勇敢な', 'es': 'valiente', 'ko': '용감한'},
        {'en': ['cowardly'], 'fr': ['lâche'], 'de': ['feige'], 'zh': ['懦弱'], 'ja': ['臆病な'], 'es': ['cobarde'], 'ko': ['겁많은']}
    ),
    (
        {'en': 'wise', 'fr': 'sage', 'de': 'weise', 'zh': '明智', 'ja': '賢い', 'es': 'sabio', 'ko': '현명한'},
        {'en': ['foolish'], 'fr': ['insensé', 'stupide'], 'de': ['dumm', 'töricht'], 'zh': ['愚蠢'], 'ja': ['愚かな'], 'es': ['tonto', 'necio'], 'ko': ['어리석은']}
    ),
    (
        {'en': 'polite', 'fr': 'poli', 'de': 'höflich', 'zh': '礼貌', 'ja': '丁寧な', 'es': 'educado', 'ko': '예의 바른'},
        {'en': ['impolite', 'rude'], 'fr': ['impoli', 'grossier'], 'de': ['unhöflich', 'grob'], 'zh': ['不礼貌'], 'ja': ['失礼な', '無礼な', '粗略な', '粗雑な'], 'es': ['descortés', 'maleducado'], 'ko': ['무례한']}
    ),
    (
        {'en': 'patient', 'fr': 'patient', 'de': 'geduldig', 'zh': '耐心', 'ja': '我慢強い', 'es': 'paciente', 'ko': '인내심 있는'},
        {'en': ['impatient'], 'fr': ['impatient'], 'de': ['ungeduldig'], 'zh': ['不耐烦'], 'ja': ['せっかちな'], 'es': ['impaciente'], 'ko': ['성급한', '참을성 없는']}
    ),
    (
        {'en': 'honest', 'fr': 'honnête', 'de': 'ehrlich', 'zh': '诚实', 'ja': '正直な', 'es': 'honesto', 'ko': '정직한'},
        {'en': ['dishonest'], 'fr': ['malhonnête'], 'de': ['unehrlich'], 'zh': ['不诚实'], 'ja': ['不正直な'], 'es': ['deshonesto'], 'ko': ['부정직한']}
    ),
    (
        {'en': 'safe', 'fr': 'sûr', 'de': 'sicher', 'zh': '安全', 'ja': '安全な', 'es': 'seguro', 'ko': '안전한'},
        {'en': ['dangerous', 'unsafe'], 'fr': ['dangereux'], 'de': ['gefährlich', 'unsicher'], 'zh': ['危险'], 'ja': ['危険な'], 'es': ['peligroso', 'inseguro'], 'ko': ['위험한']}
    ),
    (
        {'en': 'active', 'fr': 'actif', 'de': 'aktiv', 'zh': '积极', 'ja': '活動的な', 'es': 'activo', 'ko': '활동적인'},
        {'en': ['inactive', 'passive'], 'fr': ['inactif', 'passif'], 'de': ['inaktiv', 'passiv'], 'zh': ['消极'], 'ja': ['消極的な'], 'es': ['inactivo', 'pasivo'], 'ko': ['비활동적인', '소극적인']}
    ),
    (
        {'en': 'clean', 'fr': 'propre', 'de': 'sauber', 'zh': '干净', 'ja': 'きれいな', 'es': 'limpio', 'ko': '깨끗한'},
        {'en': ['dirty'], 'fr': ['sale'], 'de': ['schmutzig'], 'zh': ['脏'], 'ja': ['汚い'], 'es': ['sucio'], 'ko': ['더러운']}
    ),
    (
        {'en': 'straight', 'fr': 'droit', 'de': 'gerade', 'zh': '直', 'ja': 'まっすぐな', 'es': 'recto', 'ko': '곧은'},
        {'en': ['curved', 'bent'], 'fr': ['courbe', 'tordu'], 'de': ['gebogen', 'krumm'], 'zh': ['弯'], 'ja': ['曲がった'], 'es': ['curvo', 'doblado'], 'ko': ['굽은', '휘어진']}
    ),
    (
        {'en': 'whole', 'fr': 'entier', 'de': 'ganz', 'zh': '完整', 'ja': '全体の', 'es': 'entero', 'ko': '전체의'},
        {'en': ['part', 'broken'], 'fr': ['partiel', 'cassé'], 'de': ['teilweise', 'gebrochen'], 'zh': ['部分', '破'], 'ja': ['部分的な', '壊れた'], 'es': ['parcial', 'roto'], 'ko': ['부분적인', '부서진']}
    ),
    (
        {'en': 'early', 'fr': 'tôt', 'de': 'früh', 'zh': '早', 'ja': '早い', 'es': 'temprano', 'ko': '이른'},
        {'en': ['late'], 'fr': ['tard'], 'de': ['spät'], 'zh': ['晚'], 'ja': ['遅い'], 'es': ['tarde'], 'ko': ['늦은']}
    ),
    (
        {'en': 'strong', 'fr': 'fort', 'de': 'stark', 'zh': '强', 'ja': '強い', 'es': 'fuerte', 'ko': '강한'},
        {'en': ['weak'], 'fr': ['faible'], 'de': ['schwach'], 'zh': ['弱'], 'ja': ['弱い'], 'es': ['débil'], 'ko': ['약한']}
    ),
    (
        {'en': 'clean', 'fr': 'propre', 'de': 'sauber', 'zh': '干净', 'ja': 'きれいな', 'es': 'limpio', 'ko': '깨끗한'},
        {'en': ['dirty'], 'fr': ['sale'], 'de': ['schmutzig'], 'zh': ['脏'], 'ja': ['汚い'], 'es': ['sucio'], 'ko': ['더러운']}
    ),
    (
        {'en': 'polite', 'fr': 'poli', 'de': 'höflich', 'zh': '礼貌', 'ja': '丁寧な', 'es': 'educado', 'ko': '예의 바른'},
        {'en': ['impolite', 'rude'], 'fr': ['impoli', 'grossier'], 'de': ['unhöflich', 'grob'], 'zh': ['不礼貌'], 'ja': ['失礼な'], 'es': ['descortés', 'maleducado'], 'ko': ['무례한']}
    ),
    (
        {'en': 'true', 'fr': 'vrai', 'de': 'wahr', 'zh': '真', 'ja': '正しい', 'es': 'verdadero', 'ko': '진실한'},
        {'en': ['false', 'untrue'], 'fr': ['faux'], 'de': ['falsch', 'unwahr'], 'zh': ['假'], 'ja': ['間違った'], 'es': ['falso'], 'ko': ['거짓의']}
    ),
    (
        {'en': 'young', 'fr': 'jeune', 'de': 'jung', 'zh': '年轻', 'ja': '若い', 'es': 'joven', 'ko': '젊은'},
        {'en': ['old'], 'fr': ['vieux'], 'de': ['alt'], 'zh': ['老'], 'ja': ['老いた'], 'es': ['viejo'], 'ko': ['늙은']}
    ),
    (
        {'en': 'dry', 'fr': 'sec', 'de': 'trocken', 'zh': '干', 'ja': '乾いた', 'es': 'seco', 'ko': '마른'},
        {'en': ['wet'], 'fr': ['humide', 'mouillé'], 'de': ['nass'], 'zh': ['湿'], 'ja': ['濡れた'], 'es': ['mojado', 'húmedo'], 'ko': ['젖은']}
    ),
    (
        {'en': 'loud', 'fr': 'fort', 'de': 'laut', 'zh': '大声', 'ja': 'うるさい', 'es': 'ruidoso', 'ko': '시끄러운'},
        {'en': ['quiet', 'silent'], 'fr': ['silencieux', 'doux'], 'de': ['leise', 'still'], 'zh': ['安静', '小声'], 'ja': ['静かな'], 'es': ['tranquilo', 'silencioso'], 'ko': ['조용한']}
    ),
    (
        {'en': 'open', 'fr': 'ouvert', 'de': 'offen', 'zh': '开', 'ja': '開いた', 'es': 'abierto', 'ko': '열린'},
        {'en': ['closed'], 'fr': ['fermé'], 'de': ['geschlossen'], 'zh': ['关'], 'ja': ['閉じた'], 'es': ['cerrado'], 'ko': ['닫힌']}
    ),
    (
        {'en': 'bright', 'fr': 'brillant', 'de': 'hell', 'zh': '亮', 'ja': '明るい', 'es': 'brillante', 'ko': '밝은'},
        {'en': ['dark', 'dim'], 'fr': ['sombre', 'obscur'], 'de': ['dunkel'], 'zh': ['暗'], 'ja': ['暗い'], 'es': ['oscuro', 'opaco'], 'ko': ['어두운', '흐릿한']}
    ),
    (
        {'en': 'hard', 'fr': 'dur', 'de': 'hart', 'zh': '硬', 'ja': '硬い', 'es': 'duro', 'ko': '딱딱한'},
        {'en': ['soft'], 'fr': ['mou'], 'de': ['weich'], 'zh': ['软'], 'ja': ['柔らかい', 'やわらかい'], 'es': ['suave'], 'ko': ['부드러운']}
    ),
    (
        {'en': 'deep', 'fr': 'profond', 'de': 'tief', 'zh': '深', 'ja': '深い', 'es': 'profundo', 'ko': '깊은'},
        {'en': ['shallow'], 'fr': ['peu profond'], 'de': ['flach'], 'zh': ['浅'], 'ja': ['浅い'], 'es': ['superficial'], 'ko': ['얕은']}
    ),
    (
        {'en': 'cold', 'fr': 'froid', 'de': 'kalt', 'zh': '冷', 'ja': '寒い', 'es': 'frío', 'ko': '차가운'},
        {'en': ['hot'], 'fr': ['chaud'], 'de': ['heiß'], 'zh': ['热'], 'ja': ['暑い'], 'es': ['caliente'], 'ko': ['뜨거운']}
    ),
    (
        {'en': 'active', 'fr': 'actif', 'de': 'aktiv', 'zh': '积极', 'ja': '活動的な', 'es': 'activo', 'ko': '활동적인'},
        {'en': ['inactive'], 'fr': ['inactif'], 'de': ['inaktiv'], 'zh': ['不积极'], 'ja': ['非活動的な'], 'es': ['inactivo'], 'ko': ['비활동적인']}
    ),
    (
        {'en': 'calm', 'fr': 'calme', 'de': 'ruhig', 'zh': '平静', 'ja': '穏やかな', 'es': 'calmado', 'ko': '차분한'},
        {'en': ['agitated', 'stormy'], 'fr': ['agité', 'orageux'], 'de': ['aufgeregt', 'stürmisch'], 'zh': ['激动', '骚乱'], 'ja': ['荒れた', '興奮した'], 'es': ['agitado', 'tormentoso'], 'ko': ['격앙된', '폭풍우치는']}
    ),
    (
        {'en': 'correct', 'fr': 'correct', 'de': 'richtig', 'zh': '正确', 'ja': '正しい', 'es': 'correcto', 'ko': '정확한'},
        {'en': ['incorrect', 'wrong'], 'fr': ['incorrect', 'faux'], 'de': ['falsch'], 'zh': ['不正确', '错误'], 'ja': ['間違っている'], 'es': ['incorrecto', 'equivocado'], 'ko': ['부정확한', '틀린']}
    ),
    (
        {'en': 'clean', 'fr': 'propre', 'de': 'sauber', 'zh': '清洁', 'ja': '清潔な', 'es': 'limpio', 'ko': '깨끗한'},
        {'en': ['unclean'], 'fr': ['impropre'], 'de': ['unsauber'], 'zh': ['不清洁'], 'ja': ['不潔な'], 'es': ['sucio'], 'ko': ['더러운']}
    ),
    (
        {'en': 'complex', 'fr': 'complexe', 'de': 'komplex', 'zh': '复杂', 'ja': '複雑な', 'es': 'complejo', 'ko': '복잡한'},
        {'en': ['simple'], 'fr': ['simple'], 'de': ['einfach'], 'zh': ['简单'], 'ja': ['単純な'], 'es': ['simple', 'sencillo'], 'ko': ['간단한', '단순한']}
    ),
    (
        {'en': 'difficult', 'fr': 'difficile', 'de': 'schwierig', 'zh': '困难', 'ja': '困難な', 'es': 'difícil', 'ko': '어려운'},
        {'en': ['easy', 'simple'], 'fr': ['facile', 'simple'], 'de': ['einfach'], 'zh': ['容易', '简单'], 'ja': ['簡単な'], 'es': ['fácil', 'sencillo'], 'ko': ['쉬운', '간단한']}
    ),
    (
        {'en': 'early', 'fr': 'précoce', 'de': 'früh', 'zh': '早', 'ja': '早い', 'es': 'temprano', 'ko': '이른'},
        {'en': ['late'], 'fr': ['tardif'], 'de': ['spät'], 'zh': ['晚'], 'ja': ['遅い'], 'es': ['tarde'], 'ko': ['늦은']}
    ),
    (
        {'en': 'effective', 'fr': 'efficace', 'de': 'effektiv', 'zh': '有效', 'ja': '効果的な', 'es': 'efectivo', 'ko': '효과적인'},
        {'en': ['ineffective'], 'fr': ['inefficace'], 'de': ['ineffektiv'], 'zh': ['无效'], 'ja': ['非効果的な'], 'es': ['ineficaz'], 'ko': ['비효과적인']}
    ),
    (
        {'en': 'famous', 'fr': 'célèbre', 'de': 'berühmt', 'zh': '著名', 'ja': '有名な', 'es': 'famoso', 'ko': '유명한'},
        {'en': ['unknown', 'obscure'], 'fr': ['inconnu', 'obscur'], 'de': ['unbekannt', 'unbedeutend'], 'zh': ['无名', '不为人知'], 'ja': ['無名の', '知られていない'], 'es': ['desconocido', 'oscuro'], 'ko': ['무명의', '잘 알려지지 않은']}
    ),
    (
        {'en': 'generous', 'fr': 'généreux', 'de': 'großzügig', 'zh': '慷慨', 'ja': '寛大な', 'es': 'generoso', 'ko': '관대한'},
        {'en': ['stingy', 'mean'], 'fr': ['avare', 'mesquin'], 'de': ['geizig'], 'zh': ['吝啬'], 'ja': ['ケチな'], 'es': ['tacaño', 'malo'], 'ko': ['인색한', '못된']}
    ),
    (
        {'en': 'content', 'fr': 'content', 'de': 'zufrieden', 'zh': '高兴', 'ja': '幸せな', 'es': 'contento', 'ko': '행복한'},
        {'en': ['unhappy', 'dissatisfied'], 'fr': ['mécontent', 'insatisfait'], 'de': ['unzufrieden'], 'zh': ['不高兴', '不满意'], 'ja': ['不幸せな'], 'es': ['infeliz', 'insatisfecho'], 'ko': ['불행한', '불만족스러운']}
    ),
    (
        {'en': 'healthy', 'fr': 'sain', 'de': 'gesund', 'zh': '健康', 'ja': '健康な', 'es': 'saludable', 'ko': '건강한'},
        {'en': ['unhealthy', 'sick'], 'fr': ['malsain', 'malade'], 'de': ['ungesund', 'krank'], 'zh': ['不健康', '生病'], 'ja': ['不健康な', '病気の'], 'es': ['insalubre', 'enfermo'], 'ko': ['건강하지 않은', '아픈']}
    ),
    (
        {'en': 'high', 'fr': 'haut', 'de': 'hoch', 'zh': '高', 'ja': '高い', 'es': 'alto', 'ko': '높은'},
        {'en': ['low'], 'fr': ['bas'], 'de': ['niedrig'], 'zh': ['低'], 'ja': ['低い'], 'es': ['bajo'], 'ko': ['낮은']}
    ),
    (
        {'en': 'important', 'fr': 'important', 'de': 'wichtig', 'zh': '重要', 'ja': '重要な', 'es': 'importante', 'ko': '중요한'},
        {'en': ['unimportant', 'trivial'], 'fr': ['sans importance', 'insignifiant'], 'de': ['unwichtig', 'trivial'], 'zh': ['不重要', '琐碎'], 'ja': ['重要でない', '取るに足らない'], 'es': ['sin importancia', 'trivial'], 'ko': ['중요하지 않은', '사소한']}
    ),
    (
        {'en': 'innocent', 'fr': 'innocent', 'de': 'unschuldig', 'zh': '无辜', 'ja': '無罪の', 'es': 'inocente', 'ko': '무고한'},
        {'en': ['guilty'], 'fr': ['coupable'], 'de': ['schuldig'], 'zh': ['有罪'], 'ja': ['有罪の'], 'es': ['culpable'], 'ko': ['유죄의']}
    ),
    (
        {'en': 'known', 'fr': 'connu', 'de': 'bekannt', 'zh': '已知', 'ja': '既知の', 'es': 'conocido', 'ko': '알려진'},
        {'en': ['unknown'], 'fr': ['inconnu'], 'de': ['unbekannt'], 'zh': ['未知'], 'ja': ['未知の'], 'es': ['desconocido'], 'ko': ['알려지지 않은']}
    ),
    (
        {'en': 'light', 'fr': 'clair', 'de': 'hell', 'zh': '浅', 'ja': '薄い', 'es': 'claro', 'ko': '옅은'},
        {'en': ['dark'], 'fr': ['foncé'], 'de': ['dunkel'], 'zh': ['深'], 'ja': ['濃い', '厚い'], 'es': ['oscuro'], 'ko': ['어두운']}
    ),
    (
        {'en': 'male', 'fr': 'masculin', 'de': 'männlich', 'zh': '男性', 'ja': '男性の', 'es': 'masculino', 'ko': '남성의'},
        {'en': ['female'], 'fr': ['féminin'], 'de': ['weiblich'], 'zh': ['女性'], 'ja': ['女性の'], 'es': ['femenino'], 'ko': ['여성의']}
    ),
    (
        {'en': 'normal', 'fr': 'normal', 'de': 'normal', 'zh': '正常', 'ja': '普通の', 'es': 'normal', 'ko': '정상적인'},
        {'en': ['abnormal', 'unusual'], 'fr': ['anormal', 'inusuel'], 'de': ['abnormal', 'ungewöhnlich'], 'zh': ['异常', '不正常'], 'ja': ['異常な'], 'es': ['anormal', 'inusual'], 'ko': ['비정상적인', '이상한']}
    ),
    (
        {'en': 'old', 'fr': 'âgé', 'de': 'alt', 'zh': '老', 'ja': '高齢の', 'es': 'mayor', 'ko': '나이든'},
        {'en': ['young'], 'fr': ['jeune'], 'de': ['jung'], 'zh': ['年轻'], 'ja': ['若い'], 'es': ['joven'], 'ko': ['젊은']}
    ),
    (
        {'en': 'possible', 'fr': 'possible', 'de': 'möglich', 'zh': '可能', 'ja': '可能な', 'es': 'posible', 'ko': '가능한'},
        {'en': ['impossible'], 'fr': ['impossible'], 'de': ['unmöglich'], 'zh': ['不可能'], 'ja': ['不可能な'], 'es': ['imposible'], 'ko': ['불가능한']}
    ),
    (
        {'en': 'private', 'fr': 'privé', 'de': 'privat', 'zh': '私人', 'ja': '個人の', 'es': 'privado', 'ko': '사적인'},
        {'en': ['public'], 'fr': ['public'], 'de': ['öffentlich'], 'zh': ['公共'], 'ja': ['公共の'], 'es': ['público'], 'ko': ['공적인']}
    ),
    (
        {'en': 'right', 'fr': 'juste', 'de': 'richtig', 'zh': '对', 'ja': '正しい', 'es': 'correcto', 'ko': '올바른'},
        {'en': ['wrong'], 'fr': ['faux'], 'de': ['falsch'], 'zh': ['错'], 'ja': ['間違っている'], 'es': ['incorrecto', 'equivocado'], 'ko': ['틀린']}
    ),
    (
        {'en': 'simple', 'fr': 'simple', 'de': 'einfach', 'zh': '简单', 'ja': '簡単な', 'es': 'sencillo', 'ko': '단순한'},
        {'en': ['complex'], 'fr': ['complexe'], 'de': ['komplex'], 'zh': ['复杂'], 'ja': ['複雑な'], 'es': ['complejo'], 'ko': ['복잡한']}
    ),
    (
        {'en': 'strong', 'fr': 'fort', 'de': 'stark', 'zh': '强', 'ja': '強い', 'es': 'fuerte', 'ko': '강한'},
        {'en': ['weak'], 'fr': ['faible'], 'de': ['schwach'], 'zh': ['弱'], 'ja': ['弱い'], 'es': ['débil'], 'ko': ['약한']}
    ),
    (
        {'en': 'sweet', 'fr': 'doux', 'de': 'süß', 'zh': '甜', 'ja': '甘い', 'es': 'dulce', 'ko': '달콤한'},
        {'en': ['sour', 'bitter'], 'fr': ['acide', 'amer'], 'de': ['sauer', 'bitter'], 'zh': ['酸', '苦'], 'ja': ['酸っぱい', '苦い'], 'es': ['agrio', 'amargo'], 'ko': ['신', '쓴']}
    ),
    (
        {'en': 'ugly', 'fr': 'laid', 'de': 'hässlich', 'zh': '丑', 'ja': '醜い', 'es': 'feo', 'ko': '못생긴'},
        {'en': ['beautiful'], 'fr': ['beau'], 'de': ['schön'], 'zh': ['美'], 'ja': ['美しい'], 'es': ['hermoso'], 'ko': ['아름다운']}
    ),
    (
        {'en': 'visible', 'fr': 'visible', 'de': 'sichtbar', 'zh': '可见', 'ja': '見える', 'es': 'visible', 'ko': '보이는'},
        {'en': ['invisible'], 'fr': ['invisible'], 'de': ['unsichtbar'], 'zh': ['不可见'], 'ja': ['見えない'], 'es': ['invisible'], 'ko': ['보이지 않는']}
    ),
    (
        {'en': 'warm', 'fr': 'chaud', 'de': 'warm', 'zh': '暖和', 'ja': '暖かい', 'es': 'cálido', 'ko': '따뜻한'},
        {'en': ['cool', 'cold'], 'fr': ['frais', 'froid'], 'de': ['kühl', 'kalt'], 'zh': ['凉', '冷'], 'ja': ['涼しい', '冷たい'], 'es': ['fresco', 'frío'], 'ko': ['시원한', '차가운']}
    ),
    (
        {'en': 'wet', 'fr': 'mouillé', 'de': 'nass', 'zh': '湿', 'ja': '濡れた', 'es': 'mojado', 'ko': '젖은'},
        {'en': ['dry'], 'fr': ['sec'], 'de': ['trocken'], 'zh': ['干'], 'ja': ['乾いた'], 'es': ['seco'], 'ko': ['마른']}
    ),
    (
        {'en': 'wide', 'fr': 'large', 'de': 'breit', 'zh': '宽', 'ja': '広い', 'es': 'ancho', 'ko': '넓은'},
        {'en': ['narrow'], 'fr': ['étroit'], 'de': ['eng'], 'zh': ['窄'], 'ja': ['狭い'], 'es': ['estrecho'], 'ko': ['좁은']}
    ),
    (
        {'en': 'young', 'fr': 'jeune', 'de': 'jung', 'zh': '年轻', 'ja': '若い', 'es': 'joven', 'ko': '젊은'},
        {'en': ['old'], 'fr': ['vieux', 'âgé'], 'de': ['alt'], 'zh': ['老'], 'ja': ['老いた'], 'es': ['viejo', 'mayor'], 'ko': ['늙은', '나이든']}
    ),
    (
        {'en': 'true', 'fr': 'vrai', 'de': 'wahr', 'zh': '真', 'ja': '真の', 'es': 'verdadero', 'ko': '참된'},
        {'en': ['false'], 'fr': ['faux'], 'de': ['falsch'], 'zh': ['假'], 'ja': ['偽の'], 'es': ['falso'], 'ko': ['거짓된']}
    ),
    (
        {'en': 'right', 'fr': 'droit', 'de': 'rechts', 'zh': '右', 'ja': '右', 'es': 'derecho', 'ko': '오른쪽'},
        {'en': ['left'], 'fr': ['gauche'], 'de': ['links'], 'zh': ['左'], 'ja': ['左'], 'es': ['izquierdo'], 'ko': ['왼쪽']}
    ),
    (
        {'en': 'full', 'fr': 'complet', 'de': 'vollständig', 'zh': '完整', 'ja': '完全な', 'es': 'completo', 'ko': '완전한'},
        {'en': ['incomplete'], 'fr': ['incomplet'], 'de': ['unvollständig'], 'zh': ['不完整'], 'ja': ['不完全な'], 'es': ['incompleto'], 'ko': ['불완전한']}
    ),
    (
        {'en': 'clean', 'fr': 'nettoyer', 'de': 'reinigen', 'zh': '清洁', 'ja': '清掃する', 'es': 'limpiar', 'ko': '청소하는'},
        {'en': ['dirty', 'soil'], 'fr': ['salir'], 'de': ['beschmutzen'], 'zh': ['弄脏'], 'ja': ['汚す'], 'es': ['ensuciar'], 'ko': ['더럽히는']}
    ),
    (
        {'en': 'smooth', 'fr': 'lisse', 'de': 'glatt', 'zh': '光滑', 'ja': '滑らかな', 'es': 'liso', 'ko': '매끄러운'},
        {'en': ['rough', 'bumpy'], 'fr': ['rugueux', 'bosselé'], 'de': ['rau', 'holprig'], 'zh': ['粗糙'], 'ja': ['粗い'], 'es': ['áspero', 'irregular'], 'ko': ['거친', '울퉁불퉁한']}
    ),
    (
        {'en': 'thin', 'fr': 'mince', 'de': 'dünn', 'zh': '薄', 'ja': '薄い', 'es': 'delgado', 'ko': '얇은'},
        {'en': ['thick', 'fat'], 'fr': ['épais'], 'de': ['dick'], 'zh': ['厚'], 'ja': ['厚い'], 'es': ['grueso'], 'ko': ['두꺼운']}
    ),
    (
        {'en': 'tall', 'fr': 'grand', 'de': 'groß', 'zh': '高', 'ja': '背が高い', 'es': 'alto', 'ko': '키가 큰'},
        {'en': ['short'], 'fr': ['petit'], 'de': ['klein'], 'zh': ['矮'], 'ja': ['背が低い'], 'es': ['bajo'], 'ko': ['키가 작은']}
    ),
    (
        {'en': 'male', 'fr': 'masculin', 'de': 'männlich', 'zh': '雄性', 'ja': '雄', 'es': 'masculino', 'ko': '수컷의'},
        {'en': ['female'], 'fr': ['féminin'], 'de': ['weiblich'], 'zh': ['雌性'], 'ja': ['雌'], 'es': ['femenino'], 'ko': ['암컷의']}
    ),
    (
        {'en': 'married', 'fr': 'marié', 'de': 'verheiratet', 'zh': '已婚', 'ja': '既婚の', 'es': 'casado', 'ko': '결혼한'},
        {'en': ['single', 'unmarried'], 'fr': ['célibataire'], 'de': ['ledig', 'unverheiratet'], 'zh': ['单身', '未婚'], 'ja': ['独身の', '未婚の'], 'es': ['soltero'], 'ko': ['독신의', '미혼의']}
    ),
    (
        {'en': 'optimistic', 'fr': 'optimiste', 'de': 'optimistisch', 'zh': '乐观', 'ja': '楽観的な', 'es': 'optimista', 'ko': '낙관적인'},
        {'en': ['pessimistic'], 'fr': ['pessimiste'], 'de': ['pessimistisch'], 'zh': ['悲观'], 'ja': ['悲観的な'], 'es': ['pesimista'], 'ko': ['비관적인']}
    ),
    (
        {'en': 'permanent', 'fr': 'permanent', 'de': 'permanent', 'zh': '永久', 'ja': '恒久的な', 'es': 'permanente', 'ko': '영구적인'},
        {'en': ['temporary'], 'fr': ['temporaire'], 'de': ['temporär'], 'zh': ['临时'], 'ja': ['一時的な'], 'es': ['temporal'], 'ko': ['일시적인']}
    ),
    (
        {'en': 'possible', 'fr': 'possible', 'de': 'möglich', 'zh': '可能', 'ja': '可能な', 'es': 'posible', 'ko': '가능한'},
        {'en': ['impossible'], 'fr': ['impossible'], 'de': ['unmöglich'], 'zh': ['不可能'], 'ja': ['不可能な'], 'es': ['imposible'], 'ko': ['불가능한']}
    ),
    (
        {'en': 'present', 'fr': 'présent', 'de': 'gegenwärtig', 'zh': '现在', 'ja': '現在の', 'es': 'presente', 'ko': '현재의'},
        {'en': ['absent', 'past'], 'fr': ['absent', 'passé'], 'de': ['abwesend', 'vergangen'], 'zh': ['缺席', '过去'], 'ja': ['不在の', '過去の'], 'es': ['ausente', 'pasado'], 'ko': ['부재의', '과거의']}
    ),
    (
        {'en': 'public', 'fr': 'public', 'de': 'öffentlich', 'zh': '公共', 'ja': '公共の', 'es': 'público', 'ko': '공공의'},
        {'en': ['private'], 'fr': ['privé'], 'de': ['privat'], 'zh': ['私人'], 'ja': ['個人の'], 'es': ['privado'], 'ko': ['사적인']}
    ),
    (
        {'en': 'real', 'fr': 'réel', 'de': 'echt', 'zh': '真实', 'ja': '本物の', 'es': 'real', 'ko': '실제적인'},
        {'en': ['fake', 'unreal'], 'fr': ['faux', 'irréel'], 'de': ['falsch', 'unreal'], 'zh': ['假', '虚假'], 'ja': ['偽物の', '非現実の'], 'es': ['falso', 'irreal'], 'ko': ['가짜의', '비현실적인']}
    ),
    (
        {'en': 'responsible', 'fr': 'responsable', 'de': 'verantwortlich', 'zh': '负责', 'ja': '責任', 'es': 'responsable', 'ko': '책임 있는'},
        {'en': ['irresponsible'], 'fr': ['irresponsable'], 'de': ['unverantwortlich'], 'zh': ['不负责'], 'ja': ['無責任な', '無責任'], 'es': ['irresponsable'], 'ko': ['무책임한']}
    ),
    (
        {'en': 'safe', 'fr': 'sûr', 'de': 'sicher', 'zh': '安全', 'ja': '安全な', 'es': 'seguro', 'ko': '안전한'},
        {'en': ['dangerous', 'unsafe'], 'fr': ['dangereux'], 'de': ['gefährlich', 'unsicher'], 'zh': ['危险'], 'ja': ['危険な'], 'es': ['peligroso', 'inseguro'], 'ko': ['위험한']}
    ),
    (
        {'en': 'single', 'fr': 'célibataire', 'de': 'ledig', 'zh': '单身', 'ja': '独身の', 'es': 'soltero', 'ko': '독신의'},
        {'en': ['married'], 'fr': ['marié'], 'de': ['verheiratet'], 'zh': ['已婚'], 'ja': ['既婚の', '結婚している'], 'es': ['casado'], 'ko': ['결혼한']}
    ),
    (
        {'en': 'soft', 'fr': 'doux', 'de': 'weich', 'zh': '软', 'ja': '柔らかい', 'es': 'suave', 'ko': '부드러운'},
        {'en': ['hard'], 'fr': ['dur'], 'de': ['hart'], 'zh': ['硬'], 'ja': ['硬い'], 'es': ['duro'], 'ko': ['딱딱한']}
    ),
    (
        {'en': 'sour', 'fr': 'acide', 'de': 'sauer', 'zh': '酸', 'ja': '酸っぱい', 'es': 'agrio', 'ko': '신'},
        {'en': ['sweet'], 'fr': ['doux'], 'de': ['süß'], 'zh': ['甜'], 'ja': ['甘い'], 'es': ['dulce'], 'ko': ['달콤한']}
    ),
    (
        {'en': 'strong', 'fr': 'fort', 'de': 'stark', 'zh': '强', 'ja': '強い', 'es': 'fuerte', 'ko': '강한'},
        {'en': ['weak'], 'fr': ['faible'], 'de': ['schwach'], 'zh': ['弱'], 'ja': ['弱い'], 'es': ['débil'], 'ko': ['약한']}
    ),
    (
        {'en': 'true', 'fr': 'vrai', 'de': 'wahr', 'zh': '真', 'ja': '真実の', 'es': 'verdadero', 'ko': '참된'},
        {'en': ['false', 'untrue'], 'fr': ['faux'], 'de': ['falsch', 'unwahr'], 'zh': ['假', '虚假'], 'ja': ['虚偽の'], 'es': ['falso'], 'ko': ['거짓된']}
    ),
    (
        {'en': 'useful', 'fr': 'utile', 'de': 'nützlich', 'zh': '有用', 'ja': '役に立つ', 'es': 'útil', 'ko': '유용한'},
        {'en': ['useless'], 'fr': ['inutile'], 'de': ['nutzlos'], 'zh': ['没用'], 'ja': ['役に立たない'], 'es': ['inútil'], 'ko': ['쓸모없는']}
    ),
    (
        {'en': 'vertical', 'fr': 'vertical', 'de': 'vertikal', 'zh': '垂直', 'ja': '垂直な', 'es': 'vertical', 'ko': '수직의'},
        {'en': ['horizontal'], 'fr': ['horizontal'], 'de': ['horizontal'], 'zh': ['水平'], 'ja': ['水平な'], 'es': ['horizontal'], 'ko': ['수평의']}
    ),
    (
        {'en': 'visible', 'fr': 'visible', 'de': 'sichtbar', 'zh': '可见', 'ja': '目に見える', 'es': 'visible', 'ko': '보이는'},
        {'en': ['invisible'], 'fr': ['invisible'], 'de': ['unsichtbar'], 'zh': ['不可见'], 'ja': ['目に見えない'], 'es': ['invisible'], 'ko': ['보이지 않는']}
    ),
    (
        {'en': 'well', 'fr': 'bien', 'de': 'gut', 'zh': '好', 'ja': '良い', 'es': 'bien', 'ko': '잘'},
        {'en': ['poorly'], 'fr': ['mal'], 'de': ['schlecht'], 'zh': ['差'], 'ja': ['悪い'], 'es': ['mal', 'pobremente'], 'ko': ['못', '형편없이']}
    ),
    (
        {'en': 'whole', 'fr': 'entier', 'de': 'ganz', 'zh': '完整', 'ja': '全体の', 'es': 'entero', 'ko': '전체의'},
        {'en': ['partial'], 'fr': ['partiel'], 'de': ['teilweise'], 'zh': ['部分'], 'ja': ['部分的な'], 'es': ['parcial'], 'ko': ['부분적인']}
    ),
    (
        {'en': 'wise', 'fr': 'sage', 'de': 'weise', 'zh': '明智', 'ja': '賢い', 'es': 'sabio', 'ko': '현명한'},
        {'en': ['foolish'], 'fr': ['insensé'], 'de': ['dumm'], 'zh': ['愚蠢'], 'ja': ['愚かな'], 'es': ['tonto'], 'ko': ['어리석은']}
    ),
    (
        {'en': 'winning', 'fr': 'gagnant', 'de': 'gewinnend', 'zh': '获胜', 'ja': '勝利の', 'es': 'ganador', 'ko': '승리하는'},
        {'en': ['losing'], 'fr': ['perdant'], 'de': ['verlierend'], 'zh': ['失败'], 'ja': ['敗北の'], 'es': ['perdedor'], 'ko': ['패배하는']}
    ),
]


big_data = [
    [{"en":"good","fr":"bon","de":"gut","zh":"好","ja":"良い","es":"bueno","ko":"좋은"},{"en":["bad"],"fr":["mauvais"],"de":["schlecht"],"zh":["坏"],"ja":["悪い"],"es":["malo"],"ko":["나쁜"]}],
    [{"en":"happy","fr":"heureux","de":"glücklich","zh":"开心","ja":"嬉しい","es":"feliz","ko":"행복한"},{"en":["sad","unhappy"],"fr":["triste","malheureux"],"de":["traurig","unglücklich"],"zh":["难过","不高兴"],"ja":["悲しい"],"es":["triste","infeliz"],"ko":["슬픈","불행한"]}],
    [{"en":"big","fr":"grand","de":"groß","zh":"大","ja":"大きい","es":"grande","ko":"큰"},{"en":["small"],"fr":["petit"],"de":["klein"],"zh":["小"],"ja":["小さい"],"es":["pequeño"],"ko":["작은"]}],
    [{"en":"hot","fr":"chaud","de":"heiß","zh":"热","ja":"暑い","es":"caliente","ko":"더운"},{"en":["cold"],"fr":["froid"],"de":["kalt"],"zh":["冷"],"ja":["寒い","冷たい"],"es":["frío"],"ko":["추운","차가운"]}],
    [{"en":"fast","fr":"rapide","de":"schnell","zh":"快","ja":"速い","es":"rápido","ko":"빠른"},{"en":["slow"],"fr":["lent","lente"],"de":["langsam"],"zh":["慢"],"ja":["遅い"],"es":["lento"],"ko":["느린"]}],
    [{"en":"light","fr":"léger","de":"leicht","zh":"轻","ja":"軽い","es":"ligero","ko":"가벼운"},{"en":["heavy","dark"],"fr":["lourd"],"de":["schwer"],"zh":["重"],"ja":["重い"],"es":["pesado"],"ko":["무거운"]}],
    [{"en":"easy","fr":"facile","de":"einfach","zh":"容易","ja":"簡単な","es":"fácil","ko":"쉬운"},{"en":["difficult","hard"],"fr":["difficile"],"de":["schwierig","schwer"],"zh":["难"],"ja":["難しい"],"es":["difícil"],"ko":["어려운"]}],
    [{"en":"new","fr":"nouveau","de":"neu","zh":"新","ja":"新しい","es":"nuevo","ko":"새로운"},{"en":["old"],"fr":["vieux","ancien"],"de":["alt"],"zh":["旧"],"ja":["古い"],"es":["viejo"],"ko":["오래된"]}],
    [{"en":"true","fr":"vrai","de":"wahr","zh":"真","ja":"正しい","es":"verdadero","ko":"진실한"},{"en":["false","untrue"],"fr":["faux"],"de":["falsch","unwahr"],"zh":["假"],"ja":["間違った"],"es":["falso","incorrecto"],"ko":["거짓의","틀린"]}],
    [{"en":"alive","fr":"vivant","de":"lebendig","zh":"活","ja":"生きている","es":"vivo","ko":"살아있는"},{"en":["dead"],"fr":["mort"],"de":["tot"],"zh":["死"],"ja":["死んでいる"],"es":["muerto"],"ko":["죽은"]}],
    [{"en":"full","fr":"plein","de":"voll","zh":"满","ja":"満杯","es":"lleno","ko":"가득 찬"},{"en":["empty"],"fr":["vide"],"de":["leer"],"zh":["空"],"ja":["空っぽ"],"es":["vacío"],"ko":["빈"]}],
    [{"en":"bright","fr":"brillant","de":"hell","zh":"亮","ja":"明るい","es":"brillante","ko":"밝은"},{"en":["dark","dim"],"fr":["sombre","obscur"],"de":["dunkel"],"zh":["暗"],"ja":["暗い"],"es":["oscuro","opaco"],"ko":["어두운","흐릿한"]}],
    [{"en":"strong","fr":"fort","de":"stark","zh":"强","ja":"強い","es":"fuerte","ko":"강한"},{"en":["weak"],"fr":["faible"],"de":["schwach"],"zh":["弱"],"ja":["弱い"],"es":["débil"],"ko":["약한"]}],
    [{"en":"clean","fr":"propre","de":"sauber","zh":"干净","ja":"きれいな","es":"limpio","ko":"깨끗한"},{"en":["dirty"],"fr":["sale"],"de":["schmutzig"],"zh":["脏"],"ja":["汚い"],"es":["sucio"],"ko":["더러운"]}],
    [{"en":"open","fr":"ouvert","de":"offen","zh":"开","ja":"開いた","es":"abierto","ko":"열린"},{"en":["closed"],"fr":["fermé"],"de":["geschlossen"],"zh":["关"],"ja":["閉じた"],"es":["cerrado"],"ko":["닫힌"]}],
    [{"en":"rich","fr":"riche","de":"reich","zh":"富裕","ja":"裕福な","es":"rico","ko":"부유한"},{"en":["poor"],"fr":["pauvre"],"de":["arm"],"zh":["贫穷"],"ja":["貧しい"],"es":["pobre"],"ko":["가난한"]}],
    [{"en":"beautiful","fr":"beau","de":"schön","zh":"美","ja":"美しい","es":"hermoso","ko":"아름다운"},{"en":["ugly"],"fr":["laid"],"de":["hässlich"],"zh":["丑"],"ja":["醜い"],"es":["feo"],"ko":["못생긴"]}],
    [{"en":"long","fr":"long","de":"lang","zh":"长","ja":"長い","es":"largo","ko":"긴"},{"en":["short"],"fr":["court"],"de":["kurz"],"zh":["短"],"ja":["短い"],"es":["corto"],"ko":["짧은"]}],
    [{"en":"wide","fr":"large","de":"breit","zh":"宽","ja":"広い","es":"ancho","ko":"넓은"},{"en":["narrow"],"fr":["étroit"],"de":["eng"],"zh":["窄"],"ja":["狭い"],"es":["estrecho"],"ko":["좁은"]}],
    [{"en":"hard","fr":"dur","de":"hart","zh":"硬","ja":"硬い","es":"duro","ko":"딱딱한"},{"en":["soft"],"fr":["mou"],"de":["weich"],"zh":["软"],"ja":["柔らかい"],"es":["suave"],"ko":["부드러운"]}],
    [{"en":"dry","fr":"sec","de":"trocken","zh":"干","ja":"乾いた","es":"seco","ko":"마른"},{"en":["wet","moist"],"fr":["humide","mouillé"],"de":["nass"],"zh":["湿"],"ja":["濡れた","湿った"],"es":["mojado","húmedo"],"ko":["젖은","축축한"]}],
    [{"en":"loud","fr":"fort","de":"laut","zh":"大声","ja":"うるさい","es":"ruidoso","ko":"시끄러운"},{"en":["quiet","silent"],"fr":["silencieux","doux"],"de":["leise","still"],"zh":["安静","小声"],"ja":["静かな"],"es":["tranquilo","silencioso"],"ko":["조용한","고요한"]}],
    [{"en":"early","fr":"tôt","de":"früh","zh":"早","ja":"早い","es":"temprano","ko":"이른"},{"en":["late"],"fr":["tard"],"de":["spät"],"zh":["晚"],"ja":["遅い"],"es":["tarde"],"ko":["늦은"]}],
    [{"en":"near","fr":"proche","de":"nah","zh":"近","ja":"近い","es":"cerca","ko":"가까운"},{"en":["far"],"fr":["loin"],"de":["fern","weit"],"zh":["远"],"ja":["遠い"],"es":["lejos"],"ko":["먼"]}],
    [{"en":"deep","fr":"profond","de":"tief","zh":"深","ja":"深い","es":"profundo","ko":"깊은"},{"en":["shallow"],"fr":["peu profond"],"de":["flach"],"zh":["浅"],"ja":["浅い"],"es":["superficial","poco profundo"],"ko":["얕은"]}],
    [{"en":"bad","fr":"mauvais","de":"schlecht","zh":"坏","ja":"悪い","es":"malo","ko":"나쁜"},{"en":["good"],"fr":["bon"],"de":["gut"],"zh":["好"],"ja":["良い"],"es":["bueno"],"ko":["좋은"]}],
    [{"en":"sad","fr":"triste","de":"traurig","zh":"难过","ja":"悲しい","es":"triste","ko":"슬픈"},{"en":["happy","joyful"],"fr":["heureux","joyeux"],"de":["glücklich"],"zh":["开心"],"ja":["嬉しい"],"es":["feliz","alegre"],"ko":["행복한","기쁜"]}],
    [{"en":"small","fr":"petit","de":"klein","zh":"小","ja":"小さい","es":"pequeño","ko":"작은"},{"en":["big","large"],"fr":["grand"],"de":["groß"],"zh":["大"],"ja":["大きい"],"es":["grande"],"ko":["큰"]}],
    [{"en":"cold","fr":"froid","de":"kalt","zh":"冷","ja":"寒い","es":"frío","ko":"추운"},{"en":["hot","warm"],"fr":["chaud"],"de":["heiß"],"zh":["热"],"ja":["暑い","熱い"],"es":["caliente","cálido"],"ko":["더운","따뜻한"]}],
    [{"en":"slow","fr":"lent","de":"langsam","zh":"慢","ja":"遅い","es":"lento","ko":"느린"},{"en":["fast","quick"],"fr":["rapide"],"de":["schnell"],"zh":["快"],"ja":["速い","早い"],"es":["rápido","veloz"],"ko":["빠른"]}],
    [{"en":"heavy","fr":"lourd","de":"schwer","zh":"重","ja":"重い","es":"pesado","ko":"무거운"},{"en":["light"],"fr":["léger"],"de":["leicht"],"zh":["轻"],"ja":["軽い"],"es":["ligero"],"ko":["가벼운"]}],
    [{"en":"difficult","fr":"difficile","de":"schwierig","zh":"难","ja":"難しい","es":"difícil","ko":"어려운"},{"en":["easy","simple"],"fr":["facile"],"de":["einfach"],"zh":["容易","易"],"ja":["簡単な","簡単"],"es":["fácil","sencillo"],"ko":["쉬운","간단한"]}],
    [{"en":"old","fr":"vieux","de":"alt","zh":"旧","ja":"古い","es":"viejo","ko":"오래된"},{"en":["new"],"fr":["nouveau","jeune"],"de":["neu"],"zh":["新"],"ja":["新しい"],"es":["nuevo","joven"],"ko":["새로운","젊은"]}],
    [{"en":"false","fr":"faux","de":"falsch","zh":"假","ja":"間違った","es":"falso","ko":"거짓의"},{"en":["true","correct"],"fr":["vrai","correct"],"de":["wahr","richtig"],"zh":["真"],"ja":["正しい"],"es":["verdadero","correcto"],"ko":["진실한","올바른"]}],
    [{"en":"dead","fr":"mort","de":"tot","zh":"死","ja":"死んでいる","es":"muerto","ko":"죽은"},{"en":["alive","living"],"fr":["vivant"],"de":["lebendig"],"zh":["活","生"],"ja":["生きている"],"es":["vivo","viviente"],"ko":["살아있는","생생한"]}],
    [{"en":"empty","fr":"vide","de":"leer","zh":"空","ja":"空っぽ","es":"vacío","ko":"빈"},{"en":["full"],"fr":["plein"],"de":["voll"],"zh":["满"],"ja":["満杯"],"es":["lleno"],"ko":["가득 찬"]}],
    [{"en":"dark","fr":"sombre","de":"dunkel","zh":"暗","ja":"暗い","es":"oscuro","ko":"어두운"},{"en":["bright","light"],"fr":["brillant","clair"],"de":["hell"],"zh":["亮","光明"],"ja":["明るい"],"es":["brillante","claro"],"ko":["밝은"]}],
    [{"en":"weak","fr":"faible","de":"schwach","zh":"弱","ja":"弱い","es":"débil","ko":"약한"},{"en":["strong"],"fr":["fort"],"de":["stark"],"zh":["强","强壮"],"ja":["強い","丈夫な"],"es":["fuerte"],"ko":["강한"]}],
    [{"en":"dirty","fr":"sale","de":"schmutzig","zh":"脏","ja":"汚い","es":"sucio","ko":"더러운"},{"en":["clean"],"fr":["propre"],"de":["sauber"],"zh":["干净"],"ja":["きれいな","きれい","綺麗"],"es":["limpio"],"ko":["깨끗한"]}],
    [{"en":"closed","fr":"fermé","de":"geschlossen","zh":"关","ja":"閉じた","es":"cerrado","ko":"닫힌"},{"en":["open"],"fr":["ouvert"],"de":["offen"],"zh":["开"],"ja":["開いた"],"es":["abierto"],"ko":["열린"]}],
    [{"en":"poor","fr":"pauvre","de":"arm","zh":"贫穷","ja":"貧しい","es":"pobre","ko":"가난한"},{"en":["rich","wealthy"],"fr":["riche"],"de":["reich"],"zh":["富裕"],"ja":["裕福な","豊かな"],"es":["rico","adinerado"],"ko":["부유한","풍부한"]}],
    [{"en":"ugly","fr":"laid","de":"hässlich","zh":"丑","ja":"醜い","es":"feo","ko":"못생긴"},{"en":["beautiful","pretty"],"fr":["beau","joli"],"de":["schön"],"zh":["美"],"ja":["美しい"],"es":["hermoso","bonito"],"ko":["아름다운","예쁜"]}],
    [{"en":"short","fr":"court","de":"kurz","zh":"短","ja":"短い","es":"corto","ko":"짧은"},{"en":["long"],"fr":["long"],"de":["lang"],"zh":["长"],"ja":["長い"],"es":["largo"],"ko":["긴"]}],
    [{"en":"narrow","fr":"étroit","de":"eng","zh":"窄","ja":"狭い","es":"estrecho","ko":"좁은"},{"en":["wide","broad"],"fr":["large"],"de":["breit"],"zh":["宽"],"ja":["広い"],"es":["ancho"],"ko":["넓은"]}],
    [{"en":"soft","fr":"mou","de":"weich","zh":"软","ja":"柔らかい","es":"suave","ko":"부드러운"},{"en":["hard","firm"],"fr":["dur"],"de":["hart"],"zh":["硬"],"ja":["硬い"],"es":["duro"],"ko":["딱딱한"]}],
    [{"en":"wet","fr":"humide","de":"nass","zh":"湿","ja":"濡れた","es":"mojado","ko":"젖은"},{"en":["dry"],"fr":["sec"],"de":["trocken"],"zh":["干"],"ja":["乾いた"],"es":["seco"],"ko":["마른"]}],
    [{"en":"quiet","fr":"silencieux","de":"leise","zh":"安静","ja":"静かな","es":"tranquilo","ko":"조용한"},{"en":["loud","noisy"],"fr":["fort","bruyant"],"de":["laut"],"zh":["大声","吵闹"],"ja":["うるさい"],"es":["ruidoso"],"ko":["시끄러운"]}],
    [{"en":"late","fr":"tard","de":"spät","zh":"晚","ja":"遅い","es":"tarde","ko":"늦은"},{"en":["early"],"fr":["tôt"],"de":["früh"],"zh":["早"],"ja":["早い"],"es":["temprano"],"ko":["이른"]}],
    [{"en":"far","fr":"loin","de":"fern","zh":"远","ja":"遠い","es":"lejos","ko":"먼"},{"en":["near","close"],"fr":["proche"],"de":["nah"],"zh":["近"],"ja":["近い"],"es":["cerca"],"ko":["가까운"]}],
    [{"en":"shallow","fr":"peu profond","de":"flach","zh":"浅","ja":"浅い","es":"superficial","ko":"얕은"},{"en":["deep"],"fr":["profond"],"de":["tief","hoch"],"zh":["深"],"ja":["深い"],"es":["profundo"],"ko":["깊은"]}],
    [{"en":"young","fr":"jeune","de":"jung","zh":"年轻","ja":"若い","es":"joven","ko":"젊은"},{"en":["old"],"fr":["vieux"],"de":["alt"],"zh":["老"],"ja":["老いた"],"es":["viejo"],"ko":["늙은"]}],
    [{"en":"kind","fr":"gentil","de":"freundlich","zh":"善良","ja":"親切な","es":"amable","ko":"친절한"},{"en":["unkind","mean"],"fr":["méchant"],"de":["unfreundlich"],"zh":["不善良","邪恶"],"ja":["意地悪な","不親切な","失礼な"],"es":["desagradable","malo"],"ko":["불친절한","못된"]}],
    [{"en":"brave","fr":"courageux","de":"mutig","zh":"勇敢","ja":"勇敢な","es":"valiente","ko":"용감한"},{"en":["cowardly"],"fr":["lâche"],"de":["feige"],"zh":["懦弱"],"ja":["臆病な"],"es":["cobarde"],"ko":["겁많은"]}],
    [{"en":"wise","fr":"sage","de":"weise","zh":"明智","ja":"賢い","es":"sabio","ko":"현명한"},{"en":["foolish"],"fr":["insensé","stupide"],"de":["dumm","töricht"],"zh":["愚蠢"],"ja":["愚かな"],"es":["tonto","necio"],"ko":["어리석은"]}],
    [{"en":"polite","fr":"poli","de":"höflich","zh":"礼貌","ja":"丁寧な","es":"educado","ko":"예의 바른"},{"en":["impolite","rude"],"fr":["impoli","grossier"],"de":["unhöflich","grob"],"zh":["不礼貌"],"ja":["失礼な","無礼な","粗略な","粗雑な"],"es":["descortés","maleducado"],"ko":["무례한"]}],
    [{"en":"patient","fr":"patient","de":"geduldig","zh":"耐心","ja":"我慢強い","es":"paciente","ko":"인내심 있는"},{"en":["impatient"],"fr":["impatient"],"de":["ungeduldig"],"zh":["不耐烦"],"ja":["せっかちな"],"es":["impaciente"],"ko":["성급한","참을성 없는"]}],
    [{"en":"honest","fr":"honnête","de":"ehrlich","zh":"诚实","ja":"正直な","es":"honesto","ko":"정직한"},{"en":["dishonest"],"fr":["malhonnête"],"de":["unehrlich"],"zh":["不诚实"],"ja":["不正直な"],"es":["deshonesto"],"ko":["부정직한"]}],
    [{"en":"safe","fr":"sûr","de":"sicher","zh":"安全","ja":"安全な","es":"seguro","ko":"안전한"},{"en":["dangerous","unsafe"],"fr":["dangereux"],"de":["gefährlich","unsicher"],"zh":["危险"],"ja":["危険な"],"es":["peligroso","inseguro"],"ko":["위험한"]}],
    [{"en":"active","fr":"actif","de":"aktiv","zh":"积极","ja":"活動的な","es":"activo","ko":"활동적인"},{"en":["inactive","passive"],"fr":["inactif","passif"],"de":["inaktiv","passiv"],"zh":["消极"],"ja":["消極的な"],"es":["inactivo","pasivo"],"ko":["비활동적인","소극적인"]}],
    [{"en":"straight","fr":"droit","de":"gerade","zh":"直","ja":"まっすぐな","es":"recto","ko":"곧은"},{"en":["curved","bent"],"fr":["courbe","tordu"],"de":["gebogen","krumm"],"zh":["弯"],"ja":["曲がった"],"es":["curvo","doblado"],"ko":["굽은","휘어진"]}],
    [{"en":"whole","fr":"entier","de":"ganz","zh":"完整","ja":"全体の","es":"entero","ko":"전체의"},{"en":["part","broken"],"fr":["partiel","cassé"],"de":["teilweise","gebrochen"],"zh":["部分","破"],"ja":["部分的な","壊れた"],"es":["parcial","roto"],"ko":["부분적인","부서진"]}],
    [{"en":"calm","fr":"calme","de":"ruhig","zh":"平静","ja":"穏やかな","es":"calmado","ko":"차분한"},{"en":["agitated","stormy"],"fr":["agité","orageux"],"de":["aufgeregt","stürmisch"],"zh":["激动","骚乱"],"ja":["荒れた","興奮した"],"es":["agitado","tormentoso"],"ko":["격앙된","폭풍우치는"]}],
    [{"en":"correct","fr":"correct","de":"richtig","zh":"正确","ja":"正しい","es":"correcto","ko":"정확한"},{"en":["incorrect","wrong"],"fr":["incorrect","faux"],"de":["falsch"],"zh":["不正确","错误"],"ja":["間違っている"],"es":["incorrecto","equivocado"],"ko":["부정확한","틀린"]}],
    [{"en":"complex","fr":"complexe","de":"komplex","zh":"复杂","ja":"複雑な","es":"complejo","ko":"복잡한"},{"en":["simple"],"fr":["simple"],"de":["einfach"],"zh":["简单"],"ja":["単純な"],"es":["simple","sencillo"],"ko":["간단한","단순한"]}],
    [{"en":"effective","fr":"efficace","de":"effektiv","zh":"有效","ja":"効果的な","es":"efectivo","ko":"효과적인"},{"en":["ineffective"],"fr":["inefficace"],"de":["ineffektiv"],"zh":["无效"],"ja":["非効果的な"],"es":["ineficaz"],"ko":["비효과적인"]}],
    [{"en":"famous","fr":"célèbre","de":"berühmt","zh":"著名","ja":"有名な","es":"famoso","ko":"유명한"},{"en":["unknown","obscure"],"fr":["inconnu","obscur"],"de":["unbekannt","unbedeutend"],"zh":["无名","不为人知"],"ja":["無名の","知られていない"],"es":["desconocido","oscuro"],"ko":["무명의","잘 알려지지 않은"]}],
    [{"en":"generous","fr":"généreux","de":"großzügig","zh":"慷慨","ja":"寛大な","es":"generoso","ko":"관대한"},{"en":["stingy","mean"],"fr":["avare","mesquin"],"de":["geizig"],"zh":["吝啬"],"ja":["ケチな"],"es":["tacaño","malo"],"ko":["인색한","못된"]}],
    [{"en":"content","fr":"content","de":"zufrieden","zh":"高兴","ja":"幸せな","es":"contento","ko":"행복한"},{"en":["unhappy","dissatisfied"],"fr":["mécontent","insatisfait"],"de":["unzufrieden"],"zh":["不高兴","不满意"],"ja":["不幸せな"],"es":["infeliz","insatisfecho"],"ko":["불행한","불만족스러운"]}],
    [{"en":"healthy","fr":"sain","de":"gesund","zh":"健康","ja":"健康な","es":"saludable","ko":"건강한"},{"en":["unhealthy","sick"],"fr":["malsain","malade"],"de":["ungesund","krank"],"zh":["不健康","生病"],"ja":["不健康な","病気の"],"es":["insalubre","enfermo"],"ko":["건강하지 않은","아픈"]}],
    [{"en":"high","fr":"haut","de":"hoch","zh":"高","ja":"高い","es":"alto","ko":"높은"},{"en":["low"],"fr":["bas"],"de":["niedrig"],"zh":["低"],"ja":["低い"],"es":["bajo"],"ko":["낮은"]}],
    [{"en":"important","fr":"important","de":"wichtig","zh":"重要","ja":"重要な","es":"importante","ko":"중요한"},{"en":["unimportant","trivial"],"fr":["sans importance","insignifiant"],"de":["unwichtig","trivial"],"zh":["不重要","琐碎"],"ja":["重要でない","取るに足らない"],"es":["sin importance","trivial"],"ko":["중요하지 않은","사소한"]}],
    [{"en":"innocent","fr":"innocent","de":"unschuldig","zh":"无辜","ja":"無罪の","es":"inocente","ko":"무고한"},{"en":["guilty"],"fr":["coupable"],"de":["schuldig"],"zh":["有罪"],"ja":["有罪の"],"es":["culpable"],"ko":["유죄의"]}],
    [{"en":"known","fr":"connu","de":"bekannt","zh":"已知","ja":"既知の","es":"conocido","ko":"알려진"},{"en":["unknown"],"fr":["inconnu"],"de":["unbekannt"],"zh":["未知"],"ja":["未知の"],"es":["desconocido"],"ko":["알려지지 않은"]}],
    [{"en":"male","fr":"masculin","de":"männlich","zh":"男性","ja":"男性の","es":"masculino","ko":"남성의"},{"en":["female"],"fr":["féminin"],"de":["weiblich"],"zh":["女性"],"ja":["女性の"],"es":["femenino"],"ko":["여성의"]}],
    [{"en":"normal","fr":"normal","de":"normal","zh":"正常","ja":"普通の","es":"normal","ko":"정상적인"},{"en":["abnormal","unusual"],"fr":["anormal","inusuel"],"de":["abnormal","ungewöhnlich"],"zh":["异常","不正常"],"ja":["異常な"],"es":["anormal","inusual"],"ko":["비정상적인","이상한"]}],
    [{"en":"possible","fr":"possible","de":"möglich","zh":"可能","ja":"可能な","es":"posible","ko":"가능한"},{"en":["impossible"],"fr":["impossible"],"de":["unmöglich"],"zh":["不可能"],"ja":["不可能な"],"es":["imposible"],"ko":["불가능한"]}],
    [{"en":"private","fr":"privé","de":"privat","zh":"私人","ja":"個人の","es":"privado","ko":"사적인"},{"en":["public"],"fr":["public"],"de":["öffentlich"],"zh":["公共"],"ja":["公共の"],"es":["público"],"ko":["공적인"]}],
    [{"en":"right","fr":"juste","de":"richtig","zh":"对","ja":"正しい","es":"correcto","ko":"올바른"},{"en":["wrong"],"fr":["faux"],"de":["falsch"],"zh":["错"],"ja":["間違っている"],"es":["incorrecto","equivocado"],"ko":["틀린"]}],
    [{"en":"simple","fr":"simple","de":"einfach","zh":"简单","ja":"簡単な","es":"sencillo","ko":"단순한"},{"en":["complex"],"fr":["complexe"],"de":["komplex"],"zh":["复杂"],"ja":["複雑な"],"es":["complejo"],"ko":["복잡한"]}],
    [{"en":"sweet","fr":"doux","de":"süß","zh":"甜","ja":"甘い","es":"dulce","ko":"달콤한"},{"en":["sour","bitter"],"fr":["acide","amer"],"de":["sauer","bitter"],"zh":["酸","苦"],"ja":["酸っぱい","苦い"],"es":["agrio","amargo"],"ko":["신","쓴"]}],
    [{"en":"visible","fr":"visible","de":"sichtbar","zh":"可见","ja":"見える","es":"visible","ko":"보이는"},{"en":["invisible"],"fr":["invisible"],"de":["unsichtbar"],"zh":["不可见"],"ja":["見えない"],"es":["invisible"],"ko":["보이지 않는"]}],
    [{"en":"warm","fr":"chaud","de":"warm","zh":"暖和","ja":"暖かい","es":"cálido","ko":"따뜻한"},{"en":["cool","cold"],"fr":["frais","froid"],"de":["kühl","kalt"],"zh":["凉","冷"],"ja":["涼しい","冷たい"],"es":["fresco","frío"],"ko":["시원한","차가운"]}],
    [{"en":"smooth","fr":"lisse","de":"glatt","zh":"光滑","ja":"滑らかな","es":"liso","ko":"매끄러운"},{"en":["rough","bumpy"],"fr":["rugueux","bosselé"],"de":["rau","holprig"],"zh":["粗糙"],"ja":["粗い"],"es":["áspero","irregular"],"ko":["거친","울퉁불퉁한"]}],
    [{"en":"thin","fr":"mince","de":"dünn","zh":"薄","ja":"薄い","es":"delgado","ko":"얇은"},{"en":["thick","fat"],"fr":["épais"],"de":["dick"],"zh":["厚"],"ja":["厚い"],"es":["grueso"],"ko":["두꺼운"]}],
    [{"en":"tall","fr":"grand","de":"groß","zh":"高","ja":"背が高い","es":"alto","ko":"키가 큰"},{"en":["short"],"fr":["petit"],"de":["klein"],"zh":["矮"],"ja":["背が低い"],"es":["bajo"],"ko":["키가 작은"]}],
    [{"en":"married","fr":"marié","de":"verheiratet","zh":"已婚","ja":"既婚の","es":"casado","ko":"결혼한"},{"en":["single","unmarried"],"fr":["célibataire"],"de":["ledig","unverheiratet"],"zh":["单身","未婚"],"ja":["独身の","未婚の"],"es":["soltero"],"ko":["독신의","미혼의"]}],
    [{"en":"optimistic","fr":"optimiste","de":"optimistisch","zh":"乐观","ja":"楽観的な","es":"optimista","ko":"낙관적인"},{"en":["pessimistic"],"fr":["pessimiste"],"de":["pessimistisch"],"zh":["悲观"],"ja":["悲観的な"],"es":["pesimista"],"ko":["비관적인"]}],
    [{"en":"permanent","fr":"permanent","de":"permanent","zh":"永久","ja":"恒久的な","es":"permanente","ko":"영구적인"},{"en":["temporary"],"fr":["temporaire"],"de":["temporär"],"zh":["临时"],"ja":["一時的な"],"es":["temporal"],"ko":["일시적인"]}],
    [{"en":"present","fr":"présent","de":"gegenwärtig","zh":"现在","ja":"現在の","es":"presente","ko":"현재의"},{"en":["absent","past"],"fr":["absent","passé"],"de":["abwesend","vergangen"],"zh":["缺席","过去"],"ja":["不在の","過去の"],"es":["ausente","pasado"],"ko":["부재의","과거의"]}],
    [{"en":"public","fr":"public","de":"öffentlich","zh":"公共","ja":"公共の","es":"público","ko":"공공의"},{"en":["private"],"fr":["privé"],"de":["privat"],"zh":["私人"],"ja":["個人の"],"es":["privado"],"ko":["사적인"]}],
    [{"en":"real","fr":"réel","de":"echt","zh":"真实","ja":"本物の","es":"real","ko":"실제적인"},{"en":["fake","unreal"],"fr":["faux","irréel"],"de":["falsch","unreal"],"zh":["假","虚假"],"ja":["偽物の","非現実の"],"es":["falso","irreal"],"ko":["가짜의","비현실적인"]}],
    [{"en":"responsible","fr":"responsable","de":"verantwortlich","zh":"负责","ja":"責任","es":"responsable","ko":"책임 있는"},{"en":["irresponsible"],"fr":["irresponsable"],"de":["unverantwortlich"],"zh":["不负责"],"ja":["無責任な","無責任"],"es":["irresponsable"],"ko":["무책임한"]}],
    [{"en":"single","fr":"célibataire","de":"ledig","zh":"单身","ja":"独身の","es":"soltero","ko":"독신의"},{"en":["married"],"fr":["marié"],"de":["verheiratet"],"zh":["已婚"],"ja":["既婚の","結婚している"],"es":["casado"],"ko":["결혼한"]}],
    [{"en":"sour","fr":"acide","de":"sauer","zh":"酸","ja":"酸っぱい","es":"agrio","ko":"신"},{"en":["sweet"],"fr":["doux"],"de":["süß"],"zh":["甜"],"ja":["甘い"],"es":["dulce"],"ko":["달콤한"]}],
    [{"en":"useful","fr":"utile","de":"nützlich","zh":"有用","ja":"役に立つ","es":"útil","ko":"유용한"},{"en":["useless"],"fr":["inutile"],"de":["nutzlos"],"zh":["没用"],"ja":["役に立たない"],"es":["inútil"],"ko":["쓸모없는"]}],
    [{"en":"vertical","fr":"vertical","de":"vertikal","zh":"垂直","ja":"垂直な","es":"vertical","ko":"수직의"},{"en":["horizontal"],"fr":["horizontal"],"de":["horizontal"],"zh":["水平"],"ja":["水平な"],"es":["horizontal"],"ko":["수평의"]}],
    [{"en":"well","fr":"bien","de":"gut","zh":"好","ja":"良い","es":"bien","ko":"잘"},{"en":["poorly"],"fr":["mal"],"de":["schlecht"],"zh":["差"],"ja":["悪い"],"es":["mal","pobremente"],"ko":["못","형편없이"]}],
    [{"en":"winning","fr":"gagnant","de":"gewinnend","zh":"获胜","ja":"勝利の","es":"ganador","ko":"승리하는"},{"en":["losing"],"fr":["perdant"],"de":["verlierend"],"zh":["失败"],"ja":["敗北の"],"es":["perdedor"],"ko":["패배하는"]}],
    [{"en":"unkind","fr":"méchant","de":"unfreundlich","zh":"不善良","ja":"意地悪な","es":"desagradable","ko":"불친절한"},{"en":["kind"],"fr":["gentil"],"de":["freundlich"],"zh":["善良"],"ja":["親切な"],"es":["amable"],"ko":["친절한"]}],
    [{"en":"cowardly","fr":"lâche","de":"feige","zh":"懦弱","ja":"臆病な","es":"cobarde","ko":"겁많은"},{"en":["brave"],"fr":["courageux"],"de":["mutig"],"zh":["勇敢"],"ja":["勇敢な"],"es":["valiente"],"ko":["용감한"]}]
]
