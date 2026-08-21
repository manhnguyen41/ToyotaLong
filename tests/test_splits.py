import pandas as pd

from service_parts_forecasting.data.splits import get_rolling_origins


def test_validation_and_test_schedules() -> None:
    dates = pd.date_range("2017-08-01", "2022-09-01", freq="MS")
    validation = get_rolling_origins("validation", dates)
    test = get_rolling_origins("test", dates)
    assert [item.history_end.strftime("%Y-%m") for item in validation] == [
        "2020-09",
        "2021-01",
        "2021-05",
    ]
    assert [item.target_dates[0].strftime("%Y-%m") for item in validation] == [
        "2020-10",
        "2021-02",
        "2021-06",
    ]
    assert [item.history_end.strftime("%Y-%m") for item in test] == [
        "2021-09",
        "2022-01",
        "2022-05",
    ]
    assert [item.target_dates[-1].strftime("%Y-%m") for item in test] == [
        "2022-01",
        "2022-05",
        "2022-09",
    ]

