# python3 -m venv venv
# source venv/bin/activate
from src.data.open_dota_fetcher import OpenDotaFetcher
from pprint import pprint

fetcher = OpenDotaFetcher()
matches = fetcher.fetch_public_matches()
print(len(matches))
pprint(matches[0])

# Run this to make mongodb
# docker run -d -p 27017:27017 --name mongodb -v mongodb_data:/data/db mongo:7.0