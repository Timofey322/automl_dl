from __future__ import annotations

import numpy as np

from automl_dl.models import GradientBoostingBaseline


def test_gradient_boosting_baseline_outputs_probabilities_and_quantiles() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(120, 5))
    direction = np.resize(np.array([-1, 0, 1]), 120)
    future_return = features[:, 0] * 0.01 + rng.normal(scale=0.002, size=120)
    model = GradientBoostingBaseline(max_iter=10).fit(
        features,
        direction,
        future_return,
    )

    probabilities = model.predict_probabilities(features[:4])
    quantiles = model.predict_quantiles(features[:4])

    assert probabilities.shape == (4, 3)
    assert quantiles.shape == (4, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
