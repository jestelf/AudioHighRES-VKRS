# losses.py
import torch.nn.functional as F

def mel_reconstruction_loss(pred_mel, tgt_mel):
    return F.l1_loss(pred_mel, tgt_mel)

def speaker_classification_loss(pred, tgt):
    return F.cross_entropy(pred, tgt)

def emotion_classification_loss(pred, tgt):
    return F.cross_entropy(pred, tgt)
