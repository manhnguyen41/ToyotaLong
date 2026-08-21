import os

import service_parts_forecasting


def test_cublas_workspace_is_configured() -> None:
    assert service_parts_forecasting.__version__
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}
