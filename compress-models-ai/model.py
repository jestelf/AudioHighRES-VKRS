# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dSame(nn.Conv1d):
    """
    Реализует 1D свёрточный слой с динамическим симметричным дополнением, чтобы выходной размер был равен:
      L_out = ceil(L_in / stride)
    """
    def forward(self, input):
        # Вычисляем необходимое дополнение
        input_length = input.shape[-1]
        # Вычисляем желаемую длину выхода (ceil division)
        output_length = (input_length + self.stride[0] - 1) // self.stride[0]
        # Формула для общего количества дополнения:
        pad_needed = max((output_length - 1) * self.stride[0] +
                         (self.kernel_size[0] - 1) * self.dilation[0] + 1 - input_length, 0)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        input = F.pad(input, (pad_left, pad_right))
        return super().forward(input)

class VectorQuantizer(nn.Module):
    """
    Реализация векторного квантования для VQ-VAE.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # inputs: (B, D, T) — временная ось T
        # Переставляем в (B, T, D)
        inputs = inputs.permute(0, 2, 1).contiguous()  # (B, T, D)
        input_shape = inputs.shape  # (B, T, D)
        flat_input = inputs.view(-1, self.embedding_dim)  # (B*T, D)
        
        # Вычисляем евклидовы расстояния между входами и всеми эмбеддингами
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*T, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)  # (B, T, D)
        
        # Вычисляем потери квантования
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Стрейт-тур эффект
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()  # (B, D, T)
        return loss, quantized

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        """
        VQ-VAE для аудио с произвольной длиной входа.
          in_channels: число входных каналов (обычно 1 для моно)
          latent_dim: число каналов в латентном представлении
          num_embeddings: число эмбеддингов для квантования
          commitment_cost: коэффициент обязательства
        """
        super(VQVAE, self).__init__()
        # Encoder: используем Conv1dSame для динамического дополнения
        self.encoder = nn.Sequential(
            Conv1dSame(in_channels, 128, kernel_size=4, stride=2),    # -> (B, 128, ceil(L/2))
            nn.ReLU(inplace=True),
            Conv1dSame(128, 256, kernel_size=4, stride=2),             # -> (B, 256, ceil(L/4))
            nn.ReLU(inplace=True),
            Conv1dSame(256, latent_dim, kernel_size=4, stride=2),        # -> (B, latent_dim, ceil(L/8))
            nn.ReLU(inplace=True)
        )
        # Векторное квантование
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        # Decoder: используем стандартный ConvTranspose1d
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # -> (B, 256, ?)
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),          # -> (B, 128, ?)
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, in_channels, kernel_size=4, stride=2, padding=1),    # -> (B, in_channels, ?)
            nn.Tanh()
        )
    
    def forward(self, x):
        # x: (B, 1, L) — L может быть произвольным
        orig_len = x.shape[-1]
        z = self.encoder(x)  # (B, latent_dim, T), T = ceil(L/8)
        vq_loss, quantized = self.vq(z)  # (B, latent_dim, T)
        x_recon = self.decoder(quantized)    # (B, 1, L_recon) – может отличаться от orig_len
        if x_recon.shape[-1] != orig_len:
            # Интерполируем до исходной длины
            x_recon = F.interpolate(x_recon, size=orig_len, mode='linear', align_corners=False)
        return x_recon, vq_loss, quantized

if __name__ == "__main__":
    model = VQVAE(in_channels=1, latent_dim=64, num_embeddings=512, commitment_cost=0.25)
    # Тестирование на входе произвольной длины, например, 12345 сэмплов
    x = torch.randn(1, 1, 12345)
    recon, vq_loss, quantized = model(x)
    print("Input length:", x.shape[-1])
    print("Reconstructed length:", recon.shape[-1])
    print("VQ Loss:", vq_loss.item())
