from pathlib import Path

from service_parts_forecasting.data.loader import load_actuals


def test_supplied_workbook_contract() -> None:
    workbook = next(Path("data").glob("*.xlsx"))
    data = load_actuals(workbook)
    assert len(data.part_ids) == 8_605
    assert len(data.dates) == 62
    assert data.dates[0].strftime("%Y-%m") == "2017-08"
    assert data.dates[-1].strftime("%Y-%m") == "2022-09"
    assert list(data.long.columns) == ["part_id", "date", "demand"]
    assert not data.long.isna().any().any()

