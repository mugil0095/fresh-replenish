"""Deep-learning demand recovery with PyPOTS (SAITS, TimesNet), set up like the dataset paper."""

from dataclasses import dataclass

import numpy as np
import torch
from pypots.imputation import SAITS, TimesNet


@dataclass
class DeepImputer:
    model: SAITS | TimesNet  # fitted
    cov_mean: np.ndarray  # covariate scaling learned at fit time, reused for prediction
    cov_std: np.ndarray


def _model(name: str, n_steps: int, n_features: int, epochs: int, batch_size: int, device: str):
    common = {"epochs": epochs, "batch_size": batch_size, "device": device, "verbose": False}
    if name == "saits":
        return SAITS(n_steps=n_steps, n_features=n_features, n_layers=2, d_model=64, n_heads=4,
                     d_k=16, d_v=16, d_ffn=128, dropout=0.1, **common)
    if name == "timesnet":
        return TimesNet(n_steps=n_steps, n_features=n_features, n_layers=2, top_k=3, d_model=32,
                        d_ffn=64, n_kernels=4, **common)
    raise ValueError(f"unknown model {name!r}; use 'saits' or 'timesnet'")


def _windows(sales, covariates, cov_mean, cov_std) -> np.ndarray:
    """As in the paper: 3 windows per series, each step [sales, covariates, hour / (hours - 1)]."""
    n, days, hours = sales.shape
    if days % 3:
        raise ValueError(f"days must be divisible by 3 (3 windows per series), got {days}")
    cov = (covariates.astype(np.float32) - cov_mean) / cov_std
    hour = np.arange(hours, dtype=np.float32) / (hours - 1)
    X = np.concatenate(
        [
            sales[..., None],
            np.broadcast_to(cov[:, :, None, :], (n, days, hours, cov.shape[-1])),
            np.broadcast_to(hour[None, None, :, None], (n, days, hours, 1)),
        ],
        axis=-1,
    )
    return X.astype(np.float32).reshape(n * 3, (days // 3) * hours, -1)


def fit_deep(
    sales: np.ndarray,
    covariates: np.ndarray,
    model: str,
    epochs: int = 10,
    seed: int = 42,
    device: str | None = None,
    batch_size: int = 32,
) -> DeepImputer:
    """Train on (series, days, hours) sales with NaN where hidden; covariates (series, days, k)."""
    flat = covariates.reshape(-1, covariates.shape[-1]).astype(np.float32)
    std = flat.std(axis=0)
    mean, std = flat.mean(axis=0), np.where(std > 0, std, 1.0)
    X = _windows(sales, covariates, mean, std)
    torch.manual_seed(seed)
    np.random.seed(seed)  # PyPOTS draws its training masks from numpy's global generator
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    imputer = _model(model, X.shape[1], X.shape[2], epochs, batch_size, device)
    imputer.fit({"X": X})
    return DeepImputer(imputer, mean, std)


def predict_deep(
    imputer: DeepImputer, sales: np.ndarray, covariates: np.ndarray, chunk: int = 64
) -> np.ndarray:
    """Fill blank hours; observed hours come back unchanged.

    PyPOTS collects every output of every batch, and SAITS returns attention maps per sample
    (480 x 480 per head), so build and predict a few series at a time (about `chunk` windows)
    and keep only the imputation; memory stays flat however many series come in.
    """
    step = max(1, chunk // 3)  # 3 windows per series
    parts = [
        imputer.model.predict(
            {"X": _windows(sales[i : i + step], covariates[i : i + step],
                           imputer.cov_mean, imputer.cov_std)}
        )["imputation"][..., 0]
        for i in range(0, len(sales), step)
    ]
    filled = np.concatenate(parts).reshape(sales.shape)
    return np.where(np.isnan(sales), filled, sales)


def deep_recover(sales, covariates, model: str, **fit_kwargs) -> np.ndarray:
    """Fit on the given sales and fill their blanks."""
    return predict_deep(fit_deep(sales, covariates, model, **fit_kwargs), sales, covariates)
