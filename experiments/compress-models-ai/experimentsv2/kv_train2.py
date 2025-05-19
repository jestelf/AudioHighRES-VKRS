import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from torchmetrics import MeanSquaredError
from torch.cuda.amp import GradScaler, autocast

EMBEDDINGS_DIR = "embeddings_dataset"


# ============================================================================
# Датасет
# ============================================================================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, limit=None):
        self.files = glob.glob(os.path.join(embeddings_dir, "*.npz"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError("В директории не найдено NPZ файлов!")

        # Определяем input_dim, открывая один файл
        with np.load(self.files[0], mmap_mode='r') as sample:
            arr1 = sample["gpt_cond_latent"]
            arr2 = sample["speaker_embedding"]
            self.input_dim = arr1.flatten().shape[0] + arr2.flatten().shape[0]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], mmap_mode='r') as data:
            arr1 = data["gpt_cond_latent"].flatten()
            arr2 = data["speaker_embedding"].flatten()
            x = np.concatenate([arr1, arr2]).astype(np.float32)
        return torch.tensor(x)


# ============================================================================
# Инвертируемый блок
# ============================================================================
class InvertibleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f = self.fc2(self.relu(self.fc1(x)))
        return x + self.scale * f

    def inverse(self, y, num_iter=3):
        x = y.clone()
        for _ in range(num_iter):
            f = self.fc2(self.relu(self.fc1(x)))
            x = y - self.scale * f
        return x


# ============================================================================
# Полная модель (с опцией reduce_dimension для уменьшения размерности)
# ============================================================================
class InvertibleAutoencoder(nn.Module):
    def __init__(self, input_dim, num_blocks=4,
                 reduce_dimension=False,
                 hidden_dim=512):
        """
        :param input_dim: исходная размерность входного вектора
        :param num_blocks: сколько блоков InvertibleBlock использовать
        :param reduce_dimension: если True, то сжимаем вход до hidden_dim (нестрогая инверсия)
        :param hidden_dim: размер скрытого вектора при reduce_dimension=True
        """
        super().__init__()
        self.reduce_dimension = reduce_dimension
        self.input_dim = input_dim

        if reduce_dimension:
            self.hidden_dim = min(input_dim, hidden_dim)
            self.pre = nn.Linear(input_dim, self.hidden_dim, bias=True)
            self.blocks = nn.ModuleList([InvertibleBlock(self.hidden_dim) for _ in range(num_blocks)])
            self.post = nn.Linear(self.hidden_dim, input_dim, bias=True)
        else:
            self.hidden_dim = input_dim
            self.blocks = nn.ModuleList([InvertibleBlock(input_dim) for _ in range(num_blocks)])

    def forward(self, x):
        if self.reduce_dimension:
            x = self.pre(x)
            for block in self.blocks:
                x = block(x)
            x = self.post(x)
            return x
        else:
            for block in self.blocks:
                x = block(x)
            return x

    def inverse(self, y, num_iter_per_block=3):
        if self.reduce_dimension:
            # При reduce_dimension у нас нет 100% инверсии.
            # Для примера просто заново делаем forward (потеря информации возможна).
            return self.forward(y)
        else:
            for block in reversed(self.blocks):
                y = block.inverse(y, num_iter=num_iter_per_block)
            return y


# ============================================================================
# Функция обучения
# ============================================================================
def train_invertible_autoencoder(dataset_dir,
                                 num_blocks=4,
                                 num_epochs=30,
                                 batch_size=1,
                                 lr=1e-4,
                                 limit=None,
                                 reduce_dimension=False,
                                 hidden_dim=512):
    dataset = EmbeddingDataset(dataset_dir, limit=limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    input_dim = dataset.input_dim
    print("input_dim =", input_dim)

    model = InvertibleAutoencoder(input_dim=input_dim,
                                  num_blocks=num_blocks,
                                  reduce_dimension=reduce_dimension,
                                  hidden_dim=hidden_dim).to(device)

    # Количество параметров в модели
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров в модели: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    metric = MeanSquaredError().to(device)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_metric = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Убираем device_type='cuda' для совместимости со старыми версиями PyTorch
            with autocast(enabled=(device.type == "cuda")):
                z = model.forward(batch)
                recon = model.inverse(z)
                loss = criterion(recon, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch.size(0)
            running_metric += metric(recon, batch).item() * batch.size(0)

        epoch_loss = running_loss / len(dataset)
        epoch_metric = running_metric / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.8f}, MSE: {epoch_metric:.8f}")

    save_path = os.path.join(dataset_dir, "invertible_autoencoder.pth")
    torch.save(model.state_dict(), save_path)
    print("Модель сохранена:", save_path)


# ============================================================================
# Функции для сжатия/восстановления
# ============================================================================
def compress_file(npz_file, model_path, num_blocks, device,
                  reduce_dimension=False,
                  hidden_dim=512):
    with np.load(npz_file, mmap_mode='r') as data:
        arr1 = data["gpt_cond_latent"].flatten()
        arr2 = data["speaker_embedding"].flatten()
        input_vector = np.concatenate([arr1, arr2]).astype(np.float32)

    input_dim = input_vector.shape[0]

    model = InvertibleAutoencoder(input_dim,
                                  num_blocks=num_blocks,
                                  reduce_dimension=reduce_dimension,
                                  hidden_dim=hidden_dim).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    x = torch.from_numpy(input_vector).to(device).unsqueeze(0)
    with torch.no_grad():
        z = model.forward(x)
    z = z.squeeze(0).cpu().numpy()

    output_file = os.path.splitext(npz_file)[0] + "_latent_inv.npz"
    np.savez(output_file, latent=z, input_dim=input_dim)
    return output_file


def decompress_file(latent_file, model_path, num_blocks, device,
                    reduce_dimension=False,
                    hidden_dim=512):
    data = np.load(latent_file)
    required_keys = ["latent", "input_dim"]
    if not all(k in data for k in required_keys):
        raise ValueError("Файл латентного представления должен содержать ключи 'latent' и 'input_dim'")

    latent = data["latent"]
    input_dim = int(data["input_dim"])

    model = InvertibleAutoencoder(input_dim,
                                  num_blocks=num_blocks,
                                  reduce_dimension=reduce_dimension,
                                  hidden_dim=hidden_dim).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    z = torch.from_numpy(latent).to(device).unsqueeze(0)
    with torch.no_grad():
        recon = model.inverse(z)

    recon = recon.squeeze(0).cpu().numpy()

    output_file = os.path.splitext(latent_file)[0] + "_reconstructed_inv.npz"
    np.savez(output_file, reconstructed=recon)
    return output_file


# ============================================================================
# Основной скрипт
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Запуск обучения инвертируемой модели")
    parser.add_argument("--num_blocks", type=int, default=4, help="Количество инвертируемых блоков")
    parser.add_argument("--epochs", type=int, default=30, help="Кол-во эпох обучения")
    parser.add_argument("--batch_size", type=int, default=1, help="Маленький batch_size для экономии памяти")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None, help="Лимит файлов в датасете (для отладки)")

    # Параметры уменьшения размерности
    parser.add_argument("--reduce_dimension", action="store_true",
                        help="Если указать, сжимаем вход до --hidden_dim (нестрогое инвертирование).")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Размер 'сжатия' при reduce_dimension=True")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        train_invertible_autoencoder(
            dataset_dir=EMBEDDINGS_DIR,
            num_blocks=args.num_blocks,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            limit=args.limit,
            reduce_dimension=args.reduce_dimension,
            hidden_dim=args.hidden_dim
        )
    else:
        print("Укажите флаг --train для обучения или допишите логику compress/decompress в соответствии с вашими задачами.")
