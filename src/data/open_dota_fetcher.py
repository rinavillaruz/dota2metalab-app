# Open Dota Fetcher 
import os
import requests
import time

class OpenDotaFetcher:
    BASE_URL = "https://api.opendota.com/api"

    def __init__(self, api_key=''):
        self.api_key = api_key or os.getenv('OPENDOTA_API_KEY', '')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Dota2MetaLab/1.0'
        })

    def _params(self, extra=None):
        params = {}
        if self.api_key:
            params['api_key'] = self.api_key
        if extra:
            params.update(extra)
        return params

    def fetch_public_matches(self, limit=1000, less_than_match_id=None, _retries=5):
        extra = {'min_rank': 60}
        if less_than_match_id:
            extra['less_than_match_id'] = less_than_match_id
        try:
            response = self.session.get(
                f"{self.BASE_URL}/publicMatches",
                params=self._params(extra),
                timeout=60    # increased from 30 to 60
            )
            response.raise_for_status()
            return response.json()[:limit]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 429:
                print("Rate limited by OpenDota, waiting 60s...")
                time.sleep(60)
                return self.fetch_public_matches(limit=limit, less_than_match_id=less_than_match_id, _retries=_retries)
            elif status == 404:
                return None
            elif status == 500 and _retries > 0:
                print(f"OpenDota returned {status}, retrying in 30s ({_retries} retries left)...")
                time.sleep(30)
                return self.fetch_public_matches(limit=limit, less_than_match_id=less_than_match_id, _retries=_retries - 1)
            raise
        except requests.exceptions.RequestException as e:
            if _retries > 0:
                print(f"OpenDota unreachable ({e}), retrying in 60s ({_retries} retries left)...")
                time.sleep(60)    # increased from 30 to 60
                return self.fetch_public_matches(limit=limit, less_than_match_id=less_than_match_id, _retries=_retries - 1)
            print("OpenDota unreachable after all retries - giving up")
            return None

    def fetch_match_details(self, match_id):
        try:
            response = self.session.get(
                f"{self.BASE_URL}/matches/{match_id}",
                params=self._params(),
                timeout=60    # increased from default
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(60)
                return self.fetch_match_details(match_id)
            elif e.response.status_code == 404:
                return None
            raise

    def fetch_hero_winrates(self):
        try:
            response      = self.session.get(
                f"{self.BASE_URL}/heroStats",
                params=self._params(),
                timeout=60    # increased from default
            )
            response.raise_for_status()
            stats         = response.json()
            hero_winrates = {}
            for hero in stats:
                hero_id = hero['id']
                hero_winrates[hero_id] = {
                    'win_rate':  hero['pub_win'] / hero['pub_pick'] if hero['pub_pick'] > 0 else 0.5,
                    'pick_rate': hero['pub_pick'] / 1000000,
                    'ban_rate':  hero['pro_ban'] / 1000
                }
            return hero_winrates
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(60)
                return self.fetch_hero_winrates()
            elif e.response.status_code == 404:
                return None
            raise

    def fetch_hero_matchups(self, hero_id):
        try:
            response = self.session.get(
                f"{self.BASE_URL}/heroes/{hero_id}/matchups",
                params=self._params(),
                timeout=60    # increased from default
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(60)
                return self.fetch_hero_matchups(hero_id)
            elif e.response.status_code == 404:
                return None
            raise