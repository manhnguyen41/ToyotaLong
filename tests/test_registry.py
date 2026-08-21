import torch

from service_parts_forecasting.models.base import BaseForecastModel
from service_parts_forecasting.models.registry import create_model, register_model


@register_model("test_dummy")
class DummyModel(BaseForecastModel):
    def __init__(self, *, input_size: int, prediction_length: int = 4) -> None:
        super().__init__()
        self.prediction_length = prediction_length

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return x.new_zeros((x.shape[0], self.prediction_length))


def test_registry_is_model_agnostic() -> None:
    model = create_model({"name": "test_dummy", "prediction_length": 4}, input_size=1)
    assert model(torch.ones(3, 12, 1)).shape == (3, 4)

