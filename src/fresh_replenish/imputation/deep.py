"""Deep-learning demand recovery with PyPOTS (SAITS, TimesNet), set up like the dataset paper."""

import numpy as np
import torch
from pypots.imputation import SAITS, TimesNet


def _model(name: str, n_steps: int, n_features: int, epochs: int, batch_size: int, device: str):
    common = {"epochs": epochs, "batch_size": batch_size, "device": device, "verbose": False}
    if name == "saits":
        return SAITS(n_steps=n_steps, n_features=n_features, n_layers=2, d_model=64, n_heads=4,
                     d_k=16, d_v=16, d_ffn=128, dropout=0.1, **common)
    if name == "timesnet":
        return TimesNet(n_steps=n_steps, n_features=n_features, n_layers=2, top_k=3, d_model=32,
                        d_ffn=64, n_kernels=4, **common)
    raise ValueError(f"unknown model {name!r}; use 'saits' or 'timesnet'")


def deep_recover(
    sales: np.ndarray,
    covariates: np.ndarray,
    model: str,
    epochs: int = 10,
    seed: int = 42,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Fill blank hours; observed hours come back unchanged.

    sales: (series, days, hours) with NaN where hidden; covariates: (series, days, k).
    As in the paper, each series is cut into 3 windows and every step gets
    [sales, standardised covariates, hour / (hours - 1)].
    """
    n, days, hours = sales.shape
    window = days // 3
    cov = covariates.astype(np.float32)
    flat = cov.reshape(-1, cov.shape[-1])
    std = flat.std(axis=0)
    cov = (cov - flat.mean(axis=0)) / np.where(std > 0, std, 1.0)
    hour = np.arange(hours, dtype=np.float32) / (hours - 1)
    X = np.concatenate(
        [
            sales[..., None],
            np.broadcast_to(cov[:, :, None, :], (n, days, hours, cov.shape[-1])),
            np.broadcast_to(hour[None, None, :, None], (n, days, hours, 1)),
        ],
        axis=-1,
    ).astype(np.float32).reshape(n * 3, window * hours, -1)

    torch.manual_seed(seed)
    np.random.seed(seed)  # PyPOTS draws its training masks from numpy's global generator
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    imputer = _model(model, X.shape[1], X.shape[2], epochs, batch_size, device)
    imputer.fit({"X": X})
    filled = imputer.predict({"X": X})["imputation"][..., 0].reshape(n, days, hours)
    return np.where(np.isnan(sales), filled, sales)
