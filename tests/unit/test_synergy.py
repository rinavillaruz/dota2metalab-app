import pytest
from src.api.app import get_team_synergy

class TestGetTeamSynergy:
    
    # Tests that a team with no synergy data defaults to 0.5 for all hero pairs.
    def test_unknown_pairs_default_to_0_5(self):
        team = [1,2,3,4,5]
        result = get_team_synergy(team, synergy={})
        assert result == pytest.approx(0.5)

    # Tests that a team where all hero pairs have 100% win rate returns a synergy score of 1.0.
    def test_all_pairs_known_returns_correct_average(self):
        team = [1, 2, 3, 4, 5]
        synergy = {}
        for i in range(5):
            for j in range(i + 1, 5):
                pair = tuple(sorted([team[i], team[j]]))
                synergy[pair] = {"wins": 10, "games": 10}
        result = get_team_synergy(team, synergy)
        assert result == pytest.approx(1.0)

    # Tests that when only some hero pairs have synergy data, known pairs use real scores while unknown pairs default to 0.5, and the average is calculated correctly across all 10 pairs.
    def test_mixed_known_unknown_pairs(self):
        team = [1, 2, 3, 4, 5]
        # only one pair known with 100% win rate, rest unknown (default 0.5)
        synergy = {(1, 2): {"wins": 10, "games": 10}}
        result = get_team_synergy(team, synergy)
        # 1 pair at 1.0, 9 pairs at 0.5 → (1.0 + 9*0.5) / 10 = 0.55
        assert result == pytest.approx(0.55)

    # Tests that a hero pair that exists in the synergy dictionary but has zero games played defaults to 0.5 instead of crashing with a division by zero error.
    def test_zero_games_defaults_to_0_5(self):
        team = [1, 2, 3, 4, 5]
        synergy = {(1, 2): {"wins": 0, "games": 0}}
        result = get_team_synergy(team, synergy)
        assert result == pytest.approx(0.5)