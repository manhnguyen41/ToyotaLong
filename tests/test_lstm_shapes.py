import torch

from service_parts_forecasting.models.lstm import GlobalLSTM


def test_lstm_shape_and_nonnegative_output() -> None:
    model = GlobalLSTM(
        input_size=3,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        prediction_length=4,
        nonnegative_output="softplus",
    )
    prediction = model(torch.randn(7, 12, 3))
    assert prediction.shape == (7, 4)
    assert torch.all(prediction >= 0)

