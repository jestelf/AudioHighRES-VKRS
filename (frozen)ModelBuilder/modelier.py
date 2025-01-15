import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from xttsv2.model import XTTS2Model
from xttsv2.data import text_to_token_ids

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")
torch.backends.cudnn.benchmark = True  # Оптимизация cuDNN для обучения

# Датасет для XTTS2
class XTTS2Dataset(Dataset):
    def __init__(self, jsonl_file, features_dir):
        self.records = []
        self.features_dir = features_dir

        # Чтение jsonl файла с путями и текстами
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    audio_path = os.path.join(features_dir, os.path.basename(record['audio_features']))
                    if os.path.exists(audio_path):
                        self.records.append({
                            'audio_features': audio_path,
                            'text': record['text']
                        })
                except Exception as e:
                    print(f"Ошибка при загрузке записи: {e}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        try:
            record = self.records[idx]
            audio_features = np.load(record['audio_features'])
            text = record['text']

            # Преобразование в torch.Tensor
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
            text_tokens = torch.tensor(text_to_token_ids(text), dtype=torch.long)
            return audio_features, text_tokens
        except Exception as e:
            print(f"Ошибка в записи {idx}: {e}")
            return torch.zeros(1), torch.zeros(1)

# Функция для обработки батчей
def collate_fn(batch):
    audio_features = [item[0] for item in batch]
    text_tokens = [item[1] for item in batch]
    audio_features = pad_sequence(audio_features, batch_first=True, padding_value=0)
    text_tokens = pad_sequence(text_tokens, batch_first=True, padding_value=0)
    return audio_features, text_tokens

# Основная функция
def main():
    # Пути и параметры
    jsonl_file = 'D:/TrainerModel/Dataset/podcast_large.jsonl'
    features_dir = 'D:/TrainerModel/Dataset/features'
    checkpoint_path = 'D:/XTTS-v2/xtts2_finetuned.pth'
    model_checkpoint_path = 'D:/XTTS-v2/model.pth'
    speakers_checkpoint_path = 'D:/XTTS-v2/speakers_xtts.pth'
    batch_size = 32
    validation_split = 0.2
    num_epochs = 20
    learning_rate = 1e-4

    # Датасет и DataLoader
    dataset = XTTS2Dataset(jsonl_file, features_dir)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Загрузка модели
    print("Загрузка модели XTTS2...")
    model = XTTS2Model.from_pretrained(model_checkpoint_path, speaker_checkpoint=speakers_checkpoint_path)
    model.to(device).half()  # FP16 для ускорения
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # Смешанная точность

    # Обучение
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for audio_features, text_tokens in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]"):
            audio_features, text_tokens = audio_features.to(device), text_tokens.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(audio_features, text_tokens)
                loss = criterion(outputs.view(-1, outputs.size(-1)), text_tokens.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio_features, text_tokens in val_loader:
                audio_features, text_tokens = audio_features.to(device), text_tokens.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(audio_features, text_tokens)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), text_tokens.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Эпоха {epoch + 1}: Обучение Loss={running_loss/len(train_loader):.4f} | Валидация Loss={val_loss:.4f}")

        scheduler.step()

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Модель сохранена: {checkpoint_path}")

    print("Обучение завершено.")

if __name__ == "__main__":
    main()
