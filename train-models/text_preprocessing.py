# text_preprocessing.py
import re
import nltk
import unidecode

nltk_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")

def normalize_text(txt):
    txt = unidecode.unidecode(txt.lower())
    txt = re.sub(r"[^a-zа-яё0-9\s]+", "", txt)
    tokens = nltk_tokenizer.tokenize(txt)
    return " ".join(tokens)
