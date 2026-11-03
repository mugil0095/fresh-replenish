"""Cost-based orders (spec step 6): newsvendor stock levels and a policy simulation.

The data has no prices, so a wasted unit costs 1 and a lost sale costs r, swept over RATIOS.
Each order covers one day: unsold units are wasted (one-day shelf life) and unmet demand is lost.
"""

import numpy as np

from fresh_replenish.metrics import DECILES

RATIOS = (0.5, 1.0, 2.0, 4.0, 9.0)  # lost-sale cost / waste cost
HEADLINE = 9.0  # availability-first, as quick commerce runs today (docs/decisions.md)


def critical_ratio(r: float) -> float:
    """The newsvendor service level: the chance of not running out that minimises cost."""
    return r / (1 + r)


def order_at(q: np.ndarray, ratio: float) -> np.ndarray:
    """Order at a critical ratio, interpolated between the deciles on q's last axis."""
    if not DECILES[0] - 1e-9 <= ratio <= DECILES[-1] + 1e-9:
        raise ValueError(f"critical ratio {ratio:.2f} is outside the deciles 0.1-0.9")
    k = min(int(np.searchsorted(DECILES, ratio + 1e-9)) - 1, len(DECILES) - 2)
    w = (ratio - DECILES[k]) / (DECILES[k + 1] - DECILES[k])
    return q[..., k] * (1 - w) + q[..., k + 1] * w


def simulate(order, demand, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-series sums of cost (waste + r x lost sales), order minus demand, and demand."""
    waste, lost = np.maximum(order - demand, 0), np.maximum(demand - order, 0)
    return (waste + r * lost).sum(axis=1), (order - demand).sum(axis=1), demand.sum(axis=1)
