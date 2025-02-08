# models_big.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import Config

class BigSpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, (Config.N_MEL_CHANNELS, 3), stride=(Config.N_MEL_CHANNELS, 1), padding=(0, 1))
        self.conv2 = nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, Config.SPEAKER_EMBED_DIM)

    def forward(self, mel):
        b, c, t = mel.shape
        x = mel.unsqueeze(1)
        x = self.conv1(x).squeeze(2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class BigEmotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, (Config.N_MEL_CHANNELS, 5), stride=(Config.N_MEL_CHANNELS, 1), padding=(0, 2))
        self.conv2 = nn.Conv1d(128, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, Config.EMOTION_EMBED_DIM)

    def forward(self, mel):
        b, c, t = mel.shape
        x = mel.unsqueeze(1)
        x = self.conv1(x).squeeze(2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        b, l, d = x.shape
        length = min(l, self.pe.size(1))
        return x[:, :length, :] + self.pe[:, :length, :]

class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.posenc = PositionalEncoding(d_model, max_len=2000)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lin = nn.Linear(d_model, Config.HIDDEN_DIM)

    def forward(self, x):
        mask = (x == 0)
        em = self.embed(x)
        em = self.posenc(em)
        out = self.enc(em, src_key_padding_mask=mask)
        out = self.lin(out)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, style_dim=Config.SPEAKER_EMBED_DIM + Config.EMOTION_EMBED_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.lstm1 = nn.LSTMCell(hidden_dim + style_dim + Config.N_MEL_CHANNELS, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, Config.N_MEL_CHANNELS)

    def forward(self, enc_out, style_emb, mel_tgt):
        b, n_mels, T = mel_tgt.shape
        seq_len = enc_out.size(1)
        h1 = torch.zeros(b, self.hidden_dim, device=enc_out.device)
        c1 = torch.zeros(b, self.hidden_dim, device=enc_out.device)
        h2 = torch.zeros(b, self.hidden_dim, device=enc_out.device)
        c2 = torch.zeros(b, self.hidden_dim, device=enc_out.device)
        outs = []
        prev = torch.zeros(b, Config.N_MEL_CHANNELS, device=enc_out.device)
        for t in range(T):
            idx = min(t, seq_len - 1)
            txt_vector = enc_out[:, idx, :]
            inp = torch.cat([txt_vector, style_emb, prev], dim=1)
            h1, c1 = self.lstm1(inp, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            frame = self.fc(h2)
            outs.append(frame.unsqueeze(2))
            if self.training:
                prev = mel_tgt[:, :, t]
            else:
                prev = frame
        return torch.cat(outs, dim=2)

class HiFiDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(Config.N_MEL_CHANNELS, 1, kernel_size=1)

    def forward(self, mel):
        return self.conv(mel).squeeze(1)

class BigModel(nn.Module):
    def __init__(self, vocab_size, n_spk, n_emo):
        super().__init__()
        self.spk_enc = BigSpeakerEncoder()
        self.emo_enc = BigEmotionEncoder()
        self.spk_cls = nn.Linear(Config.SPEAKER_EMBED_DIM, n_spk)
        self.emo_cls = nn.Linear(Config.EMOTION_EMBED_DIM, n_emo)
        self.txt_enc = TransformerTextEncoder(vocab_size= vocab_size, d_model=256, nhead=4, num_layers=4)
        self.dec = TransformerDecoder(hidden_dim=Config.HIDDEN_DIM,
                                      style_dim=Config.SPEAKER_EMBED_DIM + Config.EMOTION_EMBED_DIM)
        self.voc = HiFiDummy()

    def forward(self, ref_mel, txt_tokens):
        speaker_emb = self.spk_enc(ref_mel)
        emotion_emb = self.emo_enc(ref_mel)
        speaker_logit = self.spk_cls(speaker_emb)
        emotion_logit = self.emo_cls(emotion_emb)
        style_vector = torch.cat([speaker_emb, emotion_emb], dim=1)
        enc_text = self.txt_enc(txt_tokens)
        mel_out = self.dec(enc_text, style_vector, ref_mel)
        wav_out = self.voc(mel_out)
        return mel_out, wav_out, speaker_logit, emotion_logit
