"""
Autoencoder basado en Transformer para detección de anomalías no supervisada.

Cada ventana de features (espectro dB, longitud fija) se trocea en "parches" de tamaño fijo
(análogo a un Vision Transformer, pero en 1D): cada parche se proyecta a un embedding, se le suma
una codificación posicional aprendida y se procesa con un `TransformerEncoder`. Un decoder lineal
reconstruye cada parche a partir de su embedding. El modelo se entrena SOLO con ventanas sanas
minimizando el error de reconstrucción (MSE); el error de reconstrucción de una ventana nueva es
el score de anomalía: una ventana "normal" se reconstruye bien porque el modelo ha aprendido esa
distribución, una ventana anómala (patrón espectral no visto) se reconstruye peor.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class CodificacionPosicional(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features: int, patch_size: int = 64, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.patch_size = patch_size
        self.n_patches = math.ceil(n_features / patch_size)
        self.n_padded = self.n_patches * patch_size

        self.embed = nn.Linear(patch_size, d_model)
        self.pos = CodificacionPosicional(d_model, max_len=self.n_patches)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decode = nn.Linear(d_model, patch_size)

    def _a_parches(self, x):
        pad = self.n_padded - self.n_features
        if pad > 0:
            x = nn.functional.pad(x, (0, pad))
        return x.view(x.size(0), self.n_patches, self.patch_size)

    def forward(self, x):
        parches = self._a_parches(x)
        h = self.embed(parches)
        h = self.pos(h)
        h = self.encoder(h)
        recon_parches = self.decode(h)
        recon = recon_parches.view(x.size(0), self.n_padded)[:, :self.n_features]
        return recon


def entrenar_autoencoder(modelo, X_train, X_val, epochs=150, lr=1e-3, batch_size=64,
                          paciencia=15, device=None, verbose=True):
    """Entrena minimizando MSE de reconstrucción, con early stopping sobre X_val.
    Devuelve el modelo con los mejores pesos (menor loss de validación)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32).to(device)

    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    mejor_val = float("inf")
    mejor_estado = None
    esperas = 0

    n = X_train_t.size(0)
    for epoch in range(epochs):
        modelo.train()
        perm = torch.randperm(n)
        loss_epoch = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx].to(device)
            opt.zero_grad()
            recon = modelo(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            loss_epoch += loss.item() * xb.size(0)
        loss_epoch /= n

        modelo.eval()
        with torch.no_grad():
            val_recon = modelo(X_val_t)
            val_loss = loss_fn(val_recon, X_val_t).item()

        if val_loss < mejor_val:
            mejor_val = val_loss
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            esperas = 0
        else:
            esperas += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={loss_epoch:.4f}  val_loss={val_loss:.4f}")

        if esperas >= paciencia:
            if verbose:
                print(f"  early stopping en epoch {epoch} (mejor val_loss={mejor_val:.4f})")
            break

    modelo.load_state_dict(mejor_estado)
    return modelo, mejor_val


def error_reconstruccion(modelo, X, device=None, batch_size=256):
    """MSE de reconstrucción por fila (score de anomalía: mayor = mas anómalo)."""
    device = device or next(modelo.parameters()).device
    modelo.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    errores = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(device)
            recon = modelo(xb)
            err = ((recon - xb) ** 2).mean(dim=1)
            errores.append(err.cpu().numpy())
    return np.concatenate(errores)
