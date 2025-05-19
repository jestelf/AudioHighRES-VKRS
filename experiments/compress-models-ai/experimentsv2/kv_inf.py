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
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

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
        # Сохраним исходные dtypes (например, float64)
        self.dtype1 = str(arr1.dtype)
        self.dtype2 = str(arr2.dtype)

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
        # Остаточная связь для передачи исходной информации
        out = recon + x
        return out, z

def load_model(model_path, input_dim, latent_dim, device):
    """
    Создает модель AdvancedResidualAutoencoder с указанными параметрами,
    загружает веса из контрольной точки и переводит модель в режим eval.
    Важно: latent_dim должен совпадать с тем, что использовалось при обучении.
    """
    model = AdvancedResidualAutoencoder(input_dim, latent_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compress_file(npz_file, model_path, latent_dim, device):
    """
    Сжимает NPZ-файл (с ключами 'gpt_cond_latent' и 'speaker_embedding') в латентное представление.
    Сохраняет метаданные: input_dim, latent_dim, исходные формы и типы данных.
    """
    data = np.load(npz_file)
    if "gpt_cond_latent" not in data or "speaker_embedding" not in data:
        raise ValueError("NPZ файл должен содержать ключи 'gpt_cond_latent' и 'speaker_embedding'")
    arr1 = data["gpt_cond_latent"]
    arr2 = data["speaker_embedding"]
    shape1 = arr1.shape
    shape2 = arr2.shape
    # Сохраним dtype исходных массивов (например, "float64")
    dtype1 = str(arr1.dtype)
    dtype2 = str(arr2.dtype)
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()
    input_vector = np.concatenate([arr1_flat, arr2_flat]).astype(np.float32)
    input_dim = input_vector.shape[0]
    
    model = load_model(model_path, input_dim, latent_dim, device)
    
    x = torch.from_numpy(input_vector).to(device).unsqueeze(0)
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(device.type=="cuda")):
            _, latent = model(x)
    latent = latent.squeeze(0).cpu().numpy()
    
    output_file = os.path.splitext(npz_file)[0] + "_latent.npz"
    np.savez(output_file, latent=latent, input_dim=input_dim, latent_dim=latent_dim,
             shape1=shape1, shape2=shape2, dtype1=dtype1, dtype2=dtype2)
    return output_file

def decompress_file(latent_file, model_path, device):
    """
    Восстанавливает данные из файла латентного представления.
    Использует метаданные: input_dim, latent_dim, исходные формы и dtype,
    затем приводит восстановленные массивы к исходному типу.
    """
    data = np.load(latent_file)
    required_keys = ["latent", "input_dim", "latent_dim", "shape1", "shape2", "dtype1", "dtype2"]
    if not all(k in data for k in required_keys):
        raise ValueError("Файл латентного представления должен содержать ключи 'latent', 'input_dim', 'latent_dim', 'shape1', 'shape2', 'dtype1' и 'dtype2'")
    latent = data["latent"]
    input_dim = int(data["input_dim"])
    latent_dim = int(data["latent_dim"])
    shape1 = tuple(data["shape1"])
    shape2 = tuple(data["shape2"])
    dtype1 = np.dtype(data["dtype1"].item())  # преобразуем строку в np.dtype
    dtype2 = np.dtype(data["dtype2"].item())
    
    model = load_model(model_path, input_dim, latent_dim, device)
    
    z = torch.from_numpy(latent).to(device).unsqueeze(0)
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(device.type=="cuda")):
            reconstructed = model.decoder(z)
    reconstructed = reconstructed.squeeze(0).cpu().numpy()
    
    size1 = np.prod(shape1)
    size2 = np.prod(shape2)
    arr1_flat = reconstructed[:size1]
    arr2_flat = reconstructed[size1:size1+size2]
    arr1 = arr1_flat.reshape(shape1).astype(dtype1)
    arr2 = arr2_flat.reshape(shape2).astype(dtype2)
    
    output_file = os.path.splitext(latent_file)[0] + "_reconstructed.npz"
    np.savez(output_file, gpt_cond_latent=arr1, speaker_embedding=arr2)
    return output_file

#######################################
# Графический интерфейс (GUI)
#######################################

def run_gui():
    root = tk.Tk()
    root.title("Компрессор/Декомпрессор автоэнкодера")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Переменные для выбранных файлов и параметров
    mode_var = tk.StringVar(value="compress")  # "compress" или "decompress"
    input_file_var = tk.StringVar()
    model_file_var = tk.StringVar()
    # Важно: latent_dim должен соответствовать тому, что использовалось при обучении
    latent_dim_var = tk.StringVar(value="1024")  # Например, если контрольная точка обучалась с latent_dim=1024
    
    def browse_input_file():
        if mode_var.get() == "compress":
            file_path = filedialog.askopenfilename(title="Выберите NPZ файл для сжатия", filetypes=[("NPZ файлы", "*.npz")])
        else:
            file_path = filedialog.askopenfilename(title="Выберите файл с латентным представлением", filetypes=[("NPZ файлы", "*.npz")])
        if file_path:
            input_file_var.set(file_path)
    
    def browse_model_file():
        file_path = filedialog.askopenfilename(title="Выберите файл модели (PTH)", filetypes=[("PTH файлы", "*.pth")])
        if file_path:
            model_file_var.set(file_path)
    
    def run_operation():
        try:
            if mode_var.get() == "compress":
                npz_file = input_file_var.get()
                model_file = model_file_var.get()
                try:
                    latent_dim = int(latent_dim_var.get())
                except ValueError:
                    messagebox.showerror("Ошибка", "Введите корректное значение для latent_dim!")
                    return
                if not npz_file or not model_file:
                    messagebox.showerror("Ошибка", "Выберите NPZ файл и файл модели!")
                    return
                status_label.config(text="Сжатие в процессе...")
                def compress_thread():
                    try:
                        output = compress_file(npz_file, model_file, latent_dim, device)
                        status_label.config(text=f"Сжатие завершено: {output}")
                        messagebox.showinfo("Готово", f"Сжатие завершено:\n{output}")
                    except Exception as e:
                        messagebox.showerror("Ошибка", str(e))
                        status_label.config(text="Ошибка при сжатии.")
                threading.Thread(target=compress_thread).start()
            else:
                latent_file = input_file_var.get()
                model_file = model_file_var.get()
                if not latent_file or not model_file:
                    messagebox.showerror("Ошибка", "Выберите файл латентного представления и файл модели!")
                    return
                status_label.config(text="Восстановление в процессе...")
                def decompress_thread():
                    try:
                        output = decompress_file(latent_file, model_file, device)
                        status_label.config(text=f"Восстановление завершено: {output}")
                        messagebox.showinfo("Готово", f"Восстановление завершено:\n{output}")
                    except Exception as e:
                        messagebox.showerror("Ошибка", str(e))
                        status_label.config(text="Ошибка при восстановлении.")
                threading.Thread(target=decompress_thread).start()
        except Exception as ex:
            messagebox.showerror("Ошибка", str(ex))
    
    mode_frame = tk.Frame(root, padx=10, pady=10)
    mode_frame.pack(fill=tk.BOTH, expand=True)
    
    tk.Label(mode_frame, text="Выберите режим:").grid(row=0, column=0, sticky=tk.W)
    tk.Radiobutton(mode_frame, text="Сжать NPZ", variable=mode_var, value="compress").grid(row=0, column=1, sticky=tk.W)
    tk.Radiobutton(mode_frame, text="Разжать (восстановить)", variable=mode_var, value="decompress").grid(row=0, column=2, sticky=tk.W)
    
    input_label = tk.Label(mode_frame, text="NPZ файл для сжатия:")
    input_label.grid(row=1, column=0, sticky=tk.W)
    input_entry = tk.Entry(mode_frame, textvariable=input_file_var, width=50)
    input_entry.grid(row=1, column=1, columnspan=2, padx=5)
    tk.Button(mode_frame, text="Обзор", command=browse_input_file).grid(row=1, column=3, padx=5)
    
    tk.Label(mode_frame, text="Файл модели (PTH):").grid(row=2, column=0, sticky=tk.W)
    model_entry = tk.Entry(mode_frame, textvariable=model_file_var, width=50)
    model_entry.grid(row=2, column=1, columnspan=2, padx=5)
    tk.Button(mode_frame, text="Обзор", command=browse_model_file).grid(row=2, column=3, padx=5)
    
    tk.Label(mode_frame, text="Latent dim:").grid(row=3, column=0, sticky=tk.W)
    latent_dim_entry = tk.Entry(mode_frame, textvariable=latent_dim_var)
    latent_dim_entry.grid(row=3, column=1, padx=5, sticky=tk.W)
    
    def update_mode(*args):
        if mode_var.get() == "compress":
            input_label.config(text="NPZ файл для сжатия:")
            latent_dim_entry.config(state="normal")
        else:
            input_label.config(text="Файл латентного представления:")
            latent_dim_entry.config(state="disabled")
    
    mode_var.trace("w", update_mode)
    
    tk.Button(mode_frame, text="Запустить", command=run_operation).grid(row=4, column=0, columnspan=4, pady=10)
    status_label = tk.Label(mode_frame, text="Ожидание...", fg="blue")
    status_label.grid(row=5, column=0, columnspan=4)
    
    root.mainloop()

if __name__ == "__main__":
    run_gui()
