import functools

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache

from fresh_replenish.forecasting import models

pytestmark = pytest.mark.models

FOUNDATION = {
    "chronos2": (models.chronos2, models.CHRONOS2),
    "chronos2 without inputs": (
        functools.partial(models.chronos2, use_inputs=False), models.CHRONOS2
    ),
    "timesfm2.5": (models.timesfm25, models.TIMESFM25),
    "toto2": (models.toto2, models.TOTO2),
}


@pytest.fixture(scope="module", params=list(FOUNDATION))
def foundation(request):
    """Each foundation model in turn, freed afterwards so one at a time fits the 4 GB GPU."""
    forecaster, (repo, revision) = FOUNDATION[request.param]
    if not isinstance(try_to_load_from_cache(repo, "model.safetensors", revision=revision), str):
        pytest.skip(f"{repo} isn't downloaded; the forecasting run downloads it")
    yield forecaster
    models.release()


def test_forecasts_a_weekly_pattern(foundation, make_series, weekly):
    s = make_series(weekly(n=8))
    q = foundation(s, 56, 7)
    actual = s.demand[:, 56:]
    assert q.shape == (8, 9, 7)
    assert (np.diff(q, axis=1) >= 0).all()
    assert np.abs(q[:, 4] - actual).sum() / actual.sum() < 0.3


def test_foundation_model_does_not_peek(foundation, make_series, weekly, no_peeking):
    no_peeking(foundation, make_series(weekly(n=8)), origin=56)
