"""
audio_checker.py
================
Загрузка модели PatentTTSNet + функция predict(path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path

# ─────────────────────────────── гиперпараметры
SAMPLE_RATE        = 16000
N_MELS             = 80
PATCH_TIME         = 4
PATCH_FREQ         = 4
D_MODEL            = 256
NUM_HEADS          = 4
NUM_LAYERS         = 4
DIM_FEEDFORWARD    = 512
SEGMENT_SIZE       = 8
NUM_SEGMENT_LAYERS = 2
NUM_CLASSES        = 3
CLASSES = ["original", "synth_same_text", "synth_random_text"]

# ─────────────────────────────── препроцессинг
def compute_mel_spectrogram(path: str):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                         n_mels=N_MELS, power=2.0)
    mel = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel.T, dtype=torch.float32)          # [T, M]

def compute_artifact_map(mel: torch.Tensor, k: int = 9):
    x = mel.unsqueeze(0).transpose(1, 2)                     # [1,M,T]
    smooth = F.avg_pool1d(x, kernel_size=k, stride=1,
                          padding=k // 2)
    return (x - smooth).transpose(1, 2).squeeze(0)           # [T, M]

# ─────────────────────────────── модель
class PatchEmbed(nn.Module):
    def __init__(self, d_model, p_time, p_freq):
        super().__init__()
        self.p_time, self.p_freq = p_time, p_freq
        self.patch_dim = p_time * p_freq
        self.proj = nn.Linear(self.patch_dim, d_model)

    def forward(self, x):                                    # x[B, T, F]
        B, T, F = x.shape
        T = (T // self.p_time) * self.p_time
        F = (F // self.p_freq) * self.p_freq
        x = x[:, :T, :F]

        nT = T // self.p_time
        nF = F // self.p_freq
        x = (x.view(B, nT, self.p_time, nF, self.p_freq)
                .permute(0, 1, 3, 2, 4)
                .reshape(B, nT * nF, self.patch_dim))
        return self.proj(x)                                  # [B, N, D]

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, d_model, nhead, layers, ff):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, layers)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, 10000, d_model))

    def forward(self, x_mel, x_art):
        B = x_mel.size(0)
        x = torch.cat([self.cls_token.repeat(B, 1, 1), x_mel, x_art], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        out = self.transformer(x)
        return out[:, 0, :], out[:, 1:, :]                    # CLS, tokens

class SegmentLevelSelfAttention(nn.Module):
    def __init__(self, d_model, nhead=2, layers=2, ff=256):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, batch_first=True
        )
        # NB: имя должно быть segment_encoder, как в обучении
        self.segment_encoder = nn.TransformerEncoder(enc_layer, layers)

    def forward(self, tokens):                               # [B, N, D]
        B, N, D = tokens.shape
        segs = (N // SEGMENT_SIZE)
        tokens = tokens[:, :segs * SEGMENT_SIZE, :]
        seg_emb = tokens.view(B, segs, SEGMENT_SIZE, D).mean(dim=2)
        seg_out = self.segment_encoder(seg_emb)
        return seg_out.mean(dim=1)                           # [B, D]

class MultiTaskHeads(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.bin_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.multi_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        return self.bin_fc(x).squeeze(-1), self.multi_fc(x)

class PatentTTSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_embed = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.art_embed = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.transformer_enc = TransformerEncoderWrapper(
            D_MODEL, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD
        )
        self.segment_analyzer = SegmentLevelSelfAttention(
            D_MODEL, nhead=2, layers=NUM_SEGMENT_LAYERS, ff=256
        )
        self.heads = MultiTaskHeads(2 * D_MODEL, NUM_CLASSES)

    def forward(self, mel, art):
        xm = self.mel_embed(mel)
        xa = self.art_embed(art)
        cls, toks = self.transformer_enc(xm, xa)
        seg = self.segment_analyzer(toks)
        fused = torch.cat([cls, seg], dim=-1)
        return self.heads(fused)

# ─────────────────────────────── загрузка весов
MODEL = PatentTTSNet()
ckpt_path = Path("models/patent_tts_net.pth")
if not ckpt_path.exists():
    raise FileNotFoundError(f"{ckpt_path} not found")

missing, unexpected = MODEL.load_state_dict(
    torch.load(ckpt_path, map_location="cpu"),
    strict=False
)
if missing or unexpected:
    raise RuntimeError(
        f"State dict mismatch!\nMissing: {missing}\nUnexpected: {unexpected}"
    )
MODEL.eval()

# ─────────────────────────────── predict
@torch.no_grad()
def predict(path: str, max_len: int = 400) -> str:
    mel = compute_mel_spectrogram(path)
    art = compute_artifact_map(mel)

    T = mel.size(0)
    if T < max_len:
        pad = max_len - T
        mel = F.pad(mel, (0, 0, 0, pad))
        art = F.pad(art, (0, 0, 0, pad))
    else:
        mel = mel[:max_len, :]
        art = art[:max_len, :]

    mel = mel.unsqueeze(0)
    art = art.unsqueeze(0)

    log_bin, log_mul = MODEL(mel, art)
    bin_lbl = "real" if torch.sigmoid(log_bin).item() < 0.5 else "fake"
    mul_lbl = CLASSES[log_mul.argmax(dim=1).item()]
    return f"BINARY: {bin_lbl}, CLASS: {mul_lbl}"

# быстрая ручная проверка
if __name__ == "__main__":
    wav = "examples/sample.wav"
    if Path(wav).exists():
        print(predict(wav))
    else:
        print("Добавьте тестовый WAV в examples/sample.wav чтобы проверить работу.")
