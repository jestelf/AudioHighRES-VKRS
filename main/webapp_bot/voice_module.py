"""
voice_module.py – XTTS v2 | клон / синтез голоса
================================================
•  model_dir   – папка с `XTTS-v2/` (веса + config.json)       → D:/prdja
•  storage_dir – корень, где будут храниться *личные* папки    → users_emb
                  └─ <user_id>/
                     ├─ speaker_embedding_*.npz
                     └─ tts_*.wav
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ────────────────────────────────────────
# logging
# ────────────────────────────────────────
logger = logging.getLogger("voice-module")
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    level=logging.INFO,
)

# ────────────────────────────────────────
# defaults for TTS sampling
# ────────────────────────────────────────
DEFAULT_PARAMS: Dict[str, float] = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.85,
    "repetition_penalty": 2.0,
    "length_penalty": 1.0,
    "speed": 1.0,
}

# ────────────────────────────────────────
# helpers
# ────────────────────────────────────────
def _ensure_wav(src: Path, samplerate: int = 16_000) -> Path:
    """Любой входной аудиофайл → mono WAV 16 кГц / 16-бит."""
    if src.suffix.lower() == ".wav":
        return src
    dst = src.with_suffix(".wav")
    (
        AudioSegment.from_file(src)
        .set_channels(1)
        .set_frame_rate(samplerate)
        .set_sample_width(2)  # 16-bit
        .export(dst, format="wav")
    )
    return dst


def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _clamp(v: float, low: float, high: float) -> float:
    """Ограничить значение диапазоном [low, high]."""
    return max(low, min(high, v))


# ────────────────────────────────────────
# основной класс
# ────────────────────────────────────────
class VoiceModule:
    """
    Parameters
    ----------
    model_dir   : str | Path
        Папка, в которой лежит `XTTS-v2/` (config.json + веса).
    storage_dir : str | Path
        Корневая папка для пользовательских данных:
        storage_dir/<user_id>/(слепки + синтезы).
    """

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #
    def __init__(self, model_dir: str | Path, storage_dir: str | Path):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.storage_root = Path(storage_dir).expanduser().resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

        self._load_tts()

        # кеши в RAM
        self.user_params: Dict[str, Dict[str, float]] = {}
        self.user_embedding: Dict[str, Path] = {}

        logger.info("VoiceModule готов. Корень хранения: %s", self.storage_root)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def set_user_params(self, user_id: str, **overrides) -> None:
        """Сохранить / переопределить личные sampling-параметры."""
        cur = self.user_params.get(user_id, {})
        self.user_params[user_id] = {**DEFAULT_PARAMS, **cur, **overrides}

    def get_user_params(self, user_id: str) -> Dict[str, float]:
        """Вернуть готовые к работе sampling-параметры для пользователя."""
        if user_id not in self.user_params:
            self.set_user_params(user_id)
        return self.user_params[user_id]

    # ---------- 1)  слепок ------------------------------------------- #
    def create_embedding(self, audio_file: str | Path, user_id: str) -> Path:
        """
        Принять аудиофайл (голосовое) и создать `speaker_embedding_*.npz`
        в папке пользователя. Возвращает путь к npz.
        """
        user_dir = self._user_dir(user_id)
        wav_path = _ensure_wav(Path(audio_file))

        g_latent, sp_emb = self.tts.get_conditioning_latents(audio_path=[str(wav_path)])
        out_path = user_dir / f"speaker_embedding_{_now()}.npz"
        np.savez(
            out_path,
            gpt_cond_latent=g_latent.cpu().numpy(),
            speaker_embedding=sp_emb.cpu().numpy(),
        )

        self.user_embedding[user_id] = out_path
        logger.info("Слепок сохранён: %s", out_path)
        return out_path

    # ---------- 2)  синтез ------------------------------------------ #
    def synthesize(
        self,
        user_id: str,
        text: str,
        *,
        embedding_file: Optional[str | Path] = None,
        outfile: Optional[str | Path] = None,
        **params_override,
    ) -> Path:
        """
        Синтезировать `text` c помощью последнего (или указанного) слепка.
        Возвращает путь к WAV-файлу.
        """
        # параметры
        if user_id not in self.user_params:
            self.set_user_params(user_id)
        params = {**self.user_params[user_id], **params_override}

        # speed может прилететь экзотический — ограничиваем
        params["speed"] = _clamp(float(params.get("speed", 1.0)), 0.1, 3.0)

        logger.info("TTS-параметры %s → %s", user_id, params)

        # какой слепок?
        emb_path = Path(embedding_file) if embedding_file else self.user_embedding.get(user_id)
        if not emb_path or not emb_path.exists():
            raise RuntimeError(f"Слепок для пользователя {user_id!r} не найден.")

        # загрузка слепка
        data = np.load(emb_path, allow_pickle=True)
        g_latent = torch.tensor(data["gpt_cond_latent"], device=self.device)
        sp_emb   = torch.tensor(data["speaker_embedding"], device=self.device)

        # генерация
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            wav_dict = self.tts.inference(
                text, "ru", g_latent, sp_emb,
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"],
                length_penalty=params["length_penalty"],
            )

        wav_tensor = torch.as_tensor(wav_dict["wav"]).float().unsqueeze(0)
        user_dir   = self._user_dir(user_id)
        dst        = Path(outfile) if outfile else user_dir / f"tts_{_now()}.wav"
        torchaudio.save(str(dst), wav_tensor.cpu(), 24_000)

        logger.info("Синтез сохранён: %s", dst)
        return dst

    # ------------------------------------------------------------------ #
    # internal utils
    # ------------------------------------------------------------------ #
    def _load_tts(self) -> None:
        cfg_path = self.model_dir / "XTTS-v2" / "config.json"
        ckpt_dir = self.model_dir / "XTTS-v2"

        cfg = XttsConfig(); cfg.load_json(cfg_path)
        model = Xtts.init_from_config(cfg)
        model.load_checkpoint(cfg, checkpoint_dir=ckpt_dir, eval=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts: Xtts = model.to(self.device)
        logger.info("XTTS-v2 загружена (%s).", self.device.type.upper())

    def _user_dir(self, user_id: str) -> Path:
        """
        users_emb/<user_id>  – единственное место для данных пользователя.
        """
        d = self.storage_root / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d
