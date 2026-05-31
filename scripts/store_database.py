# stores matches in MongoDB, no duplicates
import os
import time
from pymongo import MongoClient
from src.data.open_dota_fetcher import OpenDotaFetcher

opendotafetcher =   OpenDotaFetcher()
MONGO_URI       =   os.getenv('MONGO_URI', 'mongodb://localhost:27017')
client          =   MongoClient(MONGO_URI)
db              =   client['dota2metalab']
collection      =   db['matches']

existing        =   collection.count_documents({})
if existing > 17000:
    print(f"Already have {existing} matches. Skipping fetch.")
    exit(0)

all_matches     =   []
last_id         =   None

for i in range(1000):
    print(f"Printing batch {i + 1}...")

    # Stop early if we have enough
    if collection.count_documents({}) >= 17000:
        print("Reached 17,000 high rank matches. Stopping.")
        break

    matches = opendotafetcher.fetch_public_matches(less_than_match_id=last_id)

    # Handle API failure gracefully — skip batch and retry next iteration
    if matches is None:
        print("OpenDota API unavailable, waiting 60s before retrying...")
        time.sleep(60)
        continue

    # Handle empty response
    if len(matches) == 0:
        print("Empty response from OpenDota, stopping.")
        break

    all_matches.extend(matches)
    last_id       = min(m['match_id'] for m in matches)
    valid_matches = [m for m in matches if m['duration'] != 0 and len(m['radiant_team']) == 5 and len(m['dire_team']) == 5]

    for match in valid_matches:
        collection.update_one(
            {'match_id': match['match_id']},
            {'$set': match},
            upsert=True
        )
        print(f"Inserted match {match['match_id']}")

    print(f"Batch {i+1}: {len(matches)} total, {len(valid_matches)} inserted")
    time.sleep(0.5)

print(f"Total fetched: {len(all_matches)}")
print(f"Total in MongoDB: {collection.count_documents({})}")