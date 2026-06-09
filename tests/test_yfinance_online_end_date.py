import pandas as pd

from tradingagents.dataflows import y_finance as yf_data


def test_get_yfin_data_online_includes_end_date(monkeypatch):
    captured = {}

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame(
                {
                    "Open": [219.0],
                    "High": [240.4],
                    "Low": [214.5],
                    "Close": [218.0],
                    "Volume": [13_949_248],
                },
                index=pd.DatetimeIndex([pd.Timestamp("2026-06-08")]),
            )

    monkeypatch.setattr(yf_data.yf, "Ticker", FakeTicker)

    output = yf_data.get_YFin_data_online("NBIS", "2026-06-03", "2026-06-08")

    assert captured["start"] == "2026-06-03"
    assert captured["end"] == "2026-06-09"
    assert "2026-06-08" in output
    assert "218.0" in output
