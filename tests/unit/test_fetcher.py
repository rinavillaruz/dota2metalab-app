import pytest
import requests
from unittest.mock import MagicMock
from src.data.open_dota_fetcher import OpenDotaFetcher
from unittest.mock import patch, MagicMock

class TestOpenDotaFetcher:
    def setup_method(self):
        self.fetcher = OpenDotaFetcher(api_key="test_key")

    # When you create a fetcher with an api key, that key should show up in the request params.
    def test_params_includes_api_key(self):
        params = self.fetcher._params()
        assert params["api_key"] == "test_key"

    # When you create a fetcher with no api key, the params should be clean — no api_key field at all.
    def test_params_no_api_key_when_empty(self):
        fetcher = OpenDotaFetcher(api_key="")
        params = fetcher._params()
        assert "api_key" not in params

    # If I ask for 10 matches, I should get back exactly 10, not all 50.
    def test_fetch_public_matches_returns_limited_results(self):
        mock_data = [{'match_id'} for i in range(50)]
        with patch.object(self.fetcher.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_data
            mock_get.return_value.raise_for_status = MagicMock()
            result = self.fetcher.fetch_public_matches(limit=10)
        assert len(result)  == 10

    # Fetcher calls session.get()
    # Fetcher calls response.raise_for_status()
    # Fake it as throwing a 404 HTTPError
    # The code catches it and returns None
    # assert result is None ✅
    def test_fetch_public_matches_returns_none_on_404(self):
        with patch.object(self.fetcher.session, "get") as mock_get:
            http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=404))
            mock_get.return_value.raise_for_status.side_effect = http_error
            result = self.fetcher.fetch_public_matches()
        assert result is None

    # Mocks mock time.sleep so the test doesn't actually wait 30 seconds, and pass _retries=0 so it fails immediately without retrying.
    def test_fetch_public_matches_returns_none_after_all_retries_exhausted(self):
        with patch.object(self.fetcher.session, "get") as mock_get, \
            patch("time.sleep"):
            mock_get.side_effect = requests.exceptions.RequestException("timeout")
            result = self.fetcher.fetch_public_matches(_retries=0)
        assert result is None

    # Tests that the win rate is correctly calculated by dividing pub_win by pub_pick for each hero.
    def test_fetch_hero_winrates_calculates_correctly(self):
        mock_stats = [
            {"id": 1, "pub_win": 500, "pub_pick": 1000, "pro_ban": 10},
            {"id": 2, "pub_win": 300, "pub_pick": 600,  "pro_ban": 5},
        ]
        with patch.object(self.fetcher.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_stats
            mock_get.return_value.raise_for_status = MagicMock()
            result = self.fetcher.fetch_hero_winrates()
        assert result[1]["win_rate"] == pytest.approx(0.5)
        assert result[2]["win_rate"] == pytest.approx(0.5)
    
    # Tests that a hero with zero picks defaults to 0.5 win rate instead of crashing with a division by zero error.
    def test_fetch_hero_winrates_zero_pick_defaults_to_0_5(self):
        mock_stats = [{"id": 1, "pub_win": 0, "pub_pick": 0, "pro_ban": 0}]
        with patch.object(self.fetcher.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_stats
            mock_get.return_value.raise_for_status = MagicMock()
            result = self.fetcher.fetch_hero_winrates()
        assert result[1]["win_rate"] == 0.5

    def test_fetch_match_details_returns_data(self):
        mock_data = {"match_id": 123, "radiant_win": True}
        with patch.object(self.fetcher.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_data
            mock_get.return_value.raise_for_status = MagicMock()
            result = self.fetcher.fetch_match_details(123)
        assert result["match_id"] == 123

    # Tests that fetching a match that doesn't exist returns None instead of crashing.
    def test_fetch_match_details_returns_none_on_404(self):
        with patch.object(self.fetcher.session, "get") as mock_get:
            http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=404))
            mock_get.return_value.raise_for_status.side_effect = http_error
            result = self.fetcher.fetch_match_details(999)
        assert result is None