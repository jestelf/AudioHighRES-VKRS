# classifier.py — обёртка над fine-tuned моделью
# UPDATED: метод analyse() → возвращает словарь label->score  (все 10 лейблов)

import asyncio
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

SAVE_PATH = r"D:\prdja\scam_classifier_finetuned"

id2label = {
    "LABEL_0": "Безопасные сообщения",
    "LABEL_1": "Родственник в беде",
    "LABEL_2": "Выигрыши/лотереи/подарки",
    "LABEL_3": "Госорганы и службы",
    "LABEL_4": "Инвестиции и заработок",
    "LABEL_5": "Курьерские и почтовые обманы",
    "LABEL_6": "Мошенники от имени банков",
    "LABEL_7": "Поддельная служба поддержки",
    "LABEL_8": "Призывы к действию",
    "LABEL_9": "Социальные схемы",
}
_PROMPT = "Определите, к какому типу мошенничества относится следующее сообщение:"


class ScamClassifier:
    def __init__(self) -> None:
        tok = AutoTokenizer.from_pretrained(SAVE_PATH)
        mdl = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)
        self._clf = pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None, device=0)

    async def analyse(self, text: str) -> dict[str, float]:
        """
        Возвращает словарь {label_name: score}.
        """
        ipt = f"{_PROMPT} {text}"
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self._clf, ipt)
        row = res[0]  # list[dict]
        return {id2label[it["label"]]: it["score"] for it in row}


@lru_cache
def get_classifier() -> ScamClassifier:
    return ScamClassifier()
