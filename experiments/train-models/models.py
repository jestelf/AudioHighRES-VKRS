# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import Config

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 128, (Config.N_MEL_CHANNELS, 3), stride=(Config.N_MEL_CHANNELS, 1), padding=(0, 1))
        self.c2 = nn.Conv1d(128, 128, 3, 1, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, Config.SPEAKER_EMBED_DIM)

    def forward(self, mel):
        b, c, t = mel.shape
        x = mel.unsqueeze(1)
        x = self.c1(x)
        x = x.squeeze(2)
        x = F.relu(self.c2(x))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

class EmotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 128, (Config.N_MEL_CHANNELS,3), stride=(Config.N_MEL_CHANNELS,1), padding=(0,1))
        self.c2 = nn.Conv1d(128,128,3,1,1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, Config.EMOTION_EMBED_DIM)

    def forward(self, mel):
        b,c,t = mel.shape
        x = mel.unsqueeze(1)
        x = self.c1(x)
        x = x.squeeze(2)
        x = F.relu(self.c2(x))
        x = self.gmp(x).squeeze(-1)
        return self.fc(x)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hid_dim*2, hid_dim)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        return self.lin(o)

class SimpleAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.q_lin = nn.Linear(hid, hid, bias=False)
        self.e_lin = nn.Linear(hid, 1, bias=False)

    def forward(self, query, enc_out):
        b,l,h = enc_out.shape
        q = self.q_lin(query).unsqueeze(1)
        e = torch.tanh(enc_out + q)
        s = self.e_lin(e).squeeze(-1)
        a = torch.softmax(s, dim=1)
        c = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        return c, a

class TacotronDecoder(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.rnn1 = nn.LSTMCell(hid+Config.N_MEL_CHANNELS, hid)
        self.attn = SimpleAttention(hid)
        self.out_lin = nn.Linear(hid*2, Config.N_MEL_CHANNELS)

    def forward(self, enc_out, style_emb, mel_tgt):
        b, nm, T = mel_tgt.shape
        hdim = enc_out.shape[2]
        h = torch.zeros(b, hdim, device=enc_out.device)
        c = torch.zeros(b, hdim, device=enc_out.device)
        out_f = []
        a_f = []
        prev = torch.zeros(b, Config.N_MEL_CHANNELS, device=enc_out.device)
        for i in range(T):
            cc, aw = self.attn(h, enc_out)
            r_in = torch.cat([prev, cc], dim=1)
            h, c = self.rnn1(r_in, (h, c))
            hc = torch.cat([h, style_emb], dim=1)
            of = self.out_lin(hc)
            out_f.append(of.unsqueeze(2))
            a_f.append(aw.unsqueeze(1))
            if self.training:
                prev = mel_tgt[:,:,i]
            else:
                prev = of
        mo = torch.cat(out_f, dim=2)
        ao = torch.cat(a_f, dim=1)
        return mo, ao

class WaveRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(Config.N_MEL_CHANNELS, 256, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, mel):
        x = mel.transpose(1,2)
        o, _ = self.rnn(x)
        o = self.fc(o).squeeze(-1)
        return o

class VoiceCloningTTS(nn.Module):
    def __init__(self, vocab_size, n_spk, n_emo):
        super().__init__()
        self.spk_enc = SpeakerEncoder()
        self.emo_enc = EmotionEncoder()
        self.spk_cls = nn.Linear(Config.SPEAKER_EMBED_DIM, n_spk)
        self.emo_cls = nn.Linear(Config.EMOTION_EMBED_DIM, n_emo)
        self.txt_enc = TextEncoder(vocab_size, Config.TEXT_EMBED_DIM, Config.HIDDEN_DIM)
        self.dec = TacotronDecoder(Config.HIDDEN_DIM)
        self.voc = WaveRNN()

    def forward(self, ref_mel, txt_tokens):
        se = self.spk_enc(ref_mel)
        ee = self.emo_enc(ref_mel)
        sl = self.spk_cls(se)
        el = self.emo_cls(ee)
        st = torch.cat([se, ee], dim=1)
        te = self.txt_enc(txt_tokens)
        m, a = self.dec(te, st, ref_mel)
        w = self.voc(m)
        return m, w, sl, el
