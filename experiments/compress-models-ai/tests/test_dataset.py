# tests/test_dataset.py
import torch
from dataset import CommonVoiceDataset

def test_dataset_loading():
    dataset = CommonVoiceDataset(split="train", segment_length=16000)
    waveform, metadata = dataset[0]
    assert isinstance(waveform, torch.Tensor)
    assert waveform.shape == (1, 16000)
    assert isinstance(metadata, dict)
