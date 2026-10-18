from __future__ import annotations

import torch

from automl_dl.models import MultiTimeframePatchTransformer


def test_multitimeframe_model_output_shapes() -> None:
    model = MultiTimeframePatchTransformer(
        context_lengths={"1m": 16, "5m": 8},
        patch_lengths={"1m": 4, "5m": 4},
        input_features=4,
        horizons=(1, 4),
        d_model=16,
        n_heads=4,
        n_layers=1,
    )

    outputs = model(
        {
            "1m": torch.randn(3, 16, 4),
            "5m": torch.randn(3, 8, 4),
        }
    )

    assert outputs["direction_logits"].shape == (3, 3)
    assert outputs["return_quantiles"].shape == (3, 2, 3)
    assert outputs["volatility"].shape == (3, 2)
    assert outputs["excursions"].shape == (3, 2, 2)
    assert outputs["timeframe_weights"].shape == (3, 2)
    torch.testing.assert_close(
        outputs["timeframe_weights"].sum(dim=1),
        torch.ones(3),
    )
