from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
from datetime import datetime
from pymongo import MongoClient
from src.data.open_dota_fetcher import OpenDotaFetcher

MONGO_URI   = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds between retries

print(f"[{datetime.utcnow()}] Starting hero stats fetch...")

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    db     = client['dota2metalab']
    client.admin.command('ping')
    print("MongoDB connected")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    sys.exit(1)

# Fetch from OpenDota with retry
print("Fetching hero stats from OpenDota...")
fetcher       = OpenDotaFetcher()
hero_winrates = None

for attempt in range(1, MAX_RETRIES + 1):
    try:
        hero_winrates = fetcher.fetch_hero_winrates()
        if hero_winrates:
            print(f"Fetched {len(hero_winrates)} heroes on attempt {attempt}")
            break
    except Exception as e:
        print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
        if attempt < MAX_RETRIES:
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            print("All retries exhausted — OpenDota is unavailable")
            print("MongoDB still has yesterday's data — skipping update")
            sys.exit(0)  # exit 0 so CronJob doesn't mark as failed

# Upsert into MongoDB
for hero_id, stats in hero_winrates.items():
    db.hero_winrates.update_one(
        {'hero_id': hero_id},
        {'$set': {
            'hero_id':    hero_id,
            'win_rate':   stats['win_rate'],
            'pick_rate':  stats['pick_rate'],
            'ban_rate':   stats['ban_rate'],
            'updated_at': datetime.utcnow()
        }},
        upsert=True
    )

print(f"[{datetime.utcnow()}] Done! Stored win rates for {len(hero_winrates)} heroes")