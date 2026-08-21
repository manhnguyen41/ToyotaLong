from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from service_parts_forecasting.data.loader import ActualsData


@pytest.fixture
def toy_actuals() -> ActualsData:
    dates = pd.date_range("2017-08-01", periods=62, freq="MS")
    values = np.vstack(
        [
            np.arange(1, 63, dtype=np.float32),
            np.arange(101, 163, dtype=np.float32),
        ]
    )
    part_ids = ("part_1", "part_2")
    long = pd.DataFrame(
        {
            "part_id": np.repeat(part_ids, len(dates)),
            "date": np.tile(dates.to_numpy(), len(part_ids)),
            "demand": values.reshape(-1),
        }
    )
    return ActualsData(long=long, part_ids=part_ids, dates=dates, values=values)

