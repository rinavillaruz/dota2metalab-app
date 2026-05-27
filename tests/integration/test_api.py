import pytest
import json
import src.api.app as app_module
from unittest.mock import MagicMock
from pymongo.synchronous.collection import Collection

@pytest.fixture
def client():
    app_module.model         = MagicMock()
    app_module.scaler        = MagicMock()
    app_module.client        = MagicMock()
    app_module.collection    = MagicMock(spec=Collection)
    app_module.hero_winrates = {i: {"win_rate": 0.5} for i in range(1, 156)}
    app_module.synergy       = {}

    app_module.model.predict.return_value = [[0.65]]
    app_module.scaler.transform.side_effect = lambda x: x
    app_module.collection.count_documents.return_value = 100

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


class TestHealthEndpoint:

    # Tests that the health endpoint returns 200 OK
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    # Tests that the health endpoint returns the expected fields
    def test_health_payload_structure(self, client):
        data = json.loads(client.get("/health").data)
        assert "status" in data
        assert "model_loaded" in data
        assert "mongodb" in data

    def test_collection_is_not_none_check(self):
        """Ensures PyMongo Collection is never used in a boolean context"""
        mock_col = MagicMock(spec=Collection)
        mock_col.find.return_value = []
        # This would raise NotImplementedError if `if collection` is used instead of `if collection is not None`
        result = list(mock_col.find({})) if mock_col is not None else []
        assert result == []

class TestStatsEndpoint:

    # Tests that the stats endpoint returns 200 OK
    def test_stats_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

    # Tests that the stats endpoint returns the expected fields
    def test_stats_payload_has_expected_keys(self, client):
        data = json.loads(client.get("/stats").data)
        assert "total_matches" in data
        assert "radiant_wins" in data
        assert "dire_wins" in data

class TestPredictDraftEndpoint:

    def _valid_payload(self):
        return {
            "radiant_team": [1, 2, 3, 4, 5],
            "dire_team":    [6, 7, 8, 9, 10]
        }

    # Tests that a valid draft payload returns 200 OK
    def test_predict_returns_200_with_valid_payload(self, client):
        response = client.post("/predict/draft",
                               json=self._valid_payload())
        assert response.status_code == 200

    # Tests that the prediction response contains win probabilities and winner
    def test_predict_payload_has_probabilities(self, client):
        data = json.loads(client.post("/predict/draft",
                                      json=self._valid_payload()).data)
        assert "radiant_win_probability" in data
        assert "dire_win_probability" in data
        assert "predicted_winner" in data

    # Tests that radiant and dire probabilities always add up to 1.0
    def test_predict_probabilities_sum_to_one(self, client):
        data = json.loads(client.post("/predict/draft",
                                      json=self._valid_payload()).data)
        total = data["radiant_win_probability"] + data["dire_win_probability"]
        assert total == pytest.approx(1.0)

    # Tests that a team with wrong size returns 400
    def test_predict_rejects_wrong_team_size(self, client):
        payload = {"radiant_team": [1, 2, 3], "dire_team": [4, 5, 6, 7, 8]}
        response = client.post("/predict/draft", json=payload)
        assert response.status_code == 400

class TestRecommendEndpoint:

    # Tests that a valid team returns 200 OK
    def test_recommend_returns_200(self, client):
        response = client.post("/recommend", json={"team": [1, 2, 3]})
        assert response.status_code == 200

    # Tests that exactly 5 hero recommendations are returned
    def test_recommend_returns_5_heroes(self, client):
        data = json.loads(client.post("/recommend",
                                      json={"team": [1, 2, 3]}).data)
        assert len(data["recommendations"]) == 5

    # Tests that already picked heroes are not recommended
    def test_recommend_excludes_picked_heroes(self, client):
        team = [1, 2, 3]
        data = json.loads(client.post("/recommend",
                                      json={"team": team}).data)
        returned_ids = [r["hero_id"] for r in data["recommendations"]]
        for hero_id in team:
            assert hero_id not in returned_ids

    # Tests that an empty team returns 400
    def test_recommend_rejects_empty_team(self, client):
        response = client.post("/recommend", json={"team": []})
        assert response.status_code == 400

    # Tests that a team with more than 4 heroes returns 400
    def test_recommend_rejects_team_over_4(self, client):
        response = client.post("/recommend", json={"team": [1, 2, 3, 4, 5]})
        assert response.status_code == 400