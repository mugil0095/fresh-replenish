"""Hide in-stock hours the way real stockouts look, to score recovery methods on known truth.

Adapted from Dingdong-Inc/frn-50k-baseline (Apache-2.0),
latent_demand_recovery/exp/data/generate_data.py: `mimic_missing`. Changes: a seeded
`np.random.Generator` replaces the global `np.random`, and an unused running total is dropped.
"""

import numpy as np


def mimic_missing(
    patch_ts: np.ndarray,
    rng: np.random.Generator,
    p: float = 0.3,
    max_missing_patch: int = 7,
    min_missing_patch: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """patch_ts: (windows, days, hours), NaN at real stockouts.

    Returns (masked, valid_idx); valid_idx marks hours that were observed and are now hidden.
    Only fully in-stock days are touched, in two ways: runs of whole days (multi-day stockouts)
    and one end of a day (sold out until close, or empty since opening).
    """
    batch_size, patch_num, patch_len = patch_ts.shape
    patch_time = np.arange(patch_len)[None, None, :]
    patch_missing_cnt = np.isnan(patch_ts).sum(axis=-1, keepdims=True)
    non_missing_idx = patch_missing_cnt == 0

    # Runs of whole days, ending inside a streak of fully in-stock days.
    sum_vec = np.zeros_like(non_missing_idx[:, 0].astype(int))
    conti_idx = np.zeros_like(non_missing_idx)
    for i in range(patch_num):
        sum_vec = np.where(non_missing_idx[:, i], sum_vec, 0) + non_missing_idx[:, i].astype(int)
        if i > max_missing_patch:
            conti_len = rng.integers(min_missing_patch, max_missing_patch + 1, size=sum_vec.shape)
            take = (conti_len < sum_vec) & (rng.random(sum_vec.shape) < p / 10)
            conti_len = np.where(take, conti_len, 0)
            conti_tmp = np.arange(batch_size * max_missing_patch).reshape(
                batch_size, max_missing_patch, 1
            )
            conti_tmp = max_missing_patch - (conti_tmp - conti_tmp[:, 0:1])
            conti_tmp = (conti_tmp <= conti_len[:, None]) & (conti_tmp > 1)
            conti_idx[:, i - max_missing_patch + 1 : i + 1] |= conti_tmp

    # One end of a day: from a random hour to close, or from opening to a random hour.
    intra_rand_idx = (rng.random(patch_missing_cnt.shape) < p) & non_missing_idx
    start = rng.integers(0, int(patch_len * (1 - p)), size=patch_missing_cnt.shape)
    front = patch_time >= start
    back = patch_time <= patch_len - start
    pick_front = rng.random(front.shape[:-1] + (1,)) <= 0.5
    intra_idx = intra_rand_idx & np.where(pick_front, front, back)

    masked = np.where(intra_idx | conti_idx, np.nan, patch_ts)
    return masked, ~np.isnan(patch_ts) & np.isnan(masked)
