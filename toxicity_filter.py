# toxicity_filter.py

import re

class ToxicityFilter:
    def __init__(self):
        self.toxic_words = {
            'english': ['fuck', 'ass', 'bitch','nigga','chutiya','gandu','madarchod','bhosdike', 'bsdk', 'loda', 'chut', 'gaand', 'muth','Bhenchod','Madarchod','Chutiya','Harami','Gand','Lund','Lunddi','Bhosadike','Kamine','Teri Ma','Teri Behen','Saala','Gandu','Bhen ke Launde','Chutiyapa' ],
            'hindi': ['बहनचोद','माँचोद','चूतिया','हरामी','गांड','लंड','लुंडी','भोसड़ीके','कमीने','तेरी माँ','तेरी बहन','साला','गांडू','बहन के लौड़े','चूतियापा']
        }
        
        self.toxic_patterns = {
            lang: re.compile('|'.join(map(re.escape, words)), re.IGNORECASE)
            for lang, words in self.toxic_words.items()
        }

    def filter_text(self, text):
        for pattern in self.toxic_patterns.values():
            text = pattern.sub(lambda x: '*' * len(x.group()), text)
        return text
