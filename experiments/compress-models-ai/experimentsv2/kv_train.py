import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import optuna
from torchmetrics import MeanSquaredError
from torch.amp import GradScaler, autocast

EMBEDDINGS_DIR = "embeddings_dataset"

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, limit=None):
        self.files = glob.glob(os.path.join(embeddings_dir, "*.npz"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError("В директории не найдено NPZ файлов!")
        sample = np.load(self.files[0])
        arr1, arr2 = sample["gpt_cond_latent"], sample["speaker_embedding"]
        self.input_dim = arr1.flatten().shape[0] + arr2.flatten().shape[0]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        arr1, arr2 = data["gpt_cond_latent"].flatten(), data["speaker_embedding"].flatten()
        return torch.tensor(np.concatenate([arr1, arr2]).astype(np.float32))

# Улучшенная архитектура автоэнкодера с остаточными связями, BatchNorm, Dropout и LeakyReLU
class AdvancedResidualAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(8192, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        # Остаточная связь помогает передать исходную информацию
        out = recon + x
        return out, z

def train_autoencoder(dataset_dir, latent_dim=256, num_epochs=100, batch_size=32, lr=1e-3, limit=None):
    # Загрузка датасета
    dataset = EmbeddingDataset(dataset_dir, limit)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    model = AdvancedResidualAutoencoder(dataset.input_dim, latent_dim).to(device)
    # Добавляем weight decay для регуляризации
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    metric = MeanSquaredError().to(device)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    # (Опционально) можно добавить lr scheduler, например ReduceLROnPlateau:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_metric = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=(device.type == "cuda")):
                recon, _ = model(batch)
                loss = criterion(recon, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * batch.size(0)
            running_metric += metric(recon, batch).item() * batch.size(0)
        
        epoch_loss = running_loss / len(dataset)
        epoch_metric = running_metric / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, MSE: {epoch_metric:.6f}")
        scheduler.step(epoch_loss)
    
    torch.save(model.state_dict(), os.path.join(dataset_dir, "advanced_residual_autoencoder_embedding.pth"))
    print("Модель сохранена.")

def objective(trial):
    latent_dim = trial.suggest_int("latent_dim", 32, 512)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    train_autoencoder(EMBEDDINGS_DIR, latent_dim=latent_dim, num_epochs=10, batch_size=32, lr=lr, limit=100)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.train:
        train_autoencoder(EMBEDDINGS_DIR, args.latent_dim, args.epochs, args.batch_size, args.lr, args.limit)
    elif args.optimize:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        print("Лучшие параметры:", study.best_params)

# Команда запуска:
# python kv_train.py --train --latent_dim 1024 --epochs 30 --batch_size 8 --lr 0.0001 --limit 800
