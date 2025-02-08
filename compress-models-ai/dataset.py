# dataset.py
import torch
from torch.utils.data import Dataset
import torchaudio
from datasets import load_dataset

class CommonVoiceRussianDataset(Dataset):
    def __init__(self, split="train", min_length=16000):
        """
        Загружает датасет Mozilla Common Voice (версия 11.0) для русского языка.
          split: 'train', 'validation' или 'test'
          min_length: минимальная длина записи в сэмплах. Если запись короче, дополняется нулями.
        """
        self.dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split=split, trust_remote_code=True)
        self.min_length = min_length
        self.target_sr = 16000

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sample_rate = audio["sampling_rate"]
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)
        if waveform.shape[0] < self.min_length:
            pad_size = self.min_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        # Используем полную запись, если она длиннее минимальной
        waveform = waveform.unsqueeze(0)  # (1, L)
        metadata = {"sentence": sample.get("sentence", "")}
        return waveform, metadata

if __name__ == "__main__":
    dataset = CommonVoiceRussianDataset(split="train", min_length=16000)
    print("Number of samples:", len(dataset))
    waveform, metadata = dataset[0]
    print("Waveform shape:", waveform.shape)
    print("Metadata:", metadata)
