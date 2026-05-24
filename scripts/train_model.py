# trains a neural network, saves model and scaler
import os
import time
import numpy as np
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from src.data.open_dota_fetcher import OpenDotaFetcher

MODEL_DIR   =   os.getenv('MODEL_DIR', 'models')
MODEL_PATH  =   f"{MODEL_DIR}/dota2_model.h5"
if os.path.exists(MODEL_PATH):
    print("Model already exists. Skipping training.")
    exit(0)

MONGO_URI   =   os.getenv('MONGO_URI', 'mongodb://localhost:27017')
client      =   MongoClient(MONGO_URI)
db          =   client['dota2metalab']
collection  =   db['matches']

# Data Collection / Data Loading
matches     =   list(collection.find({}))
X           =   []
y           =   []

# Build synergy matrix from existing matches
print("Building synergy matrix...")
synergy = {}
for match in matches:
    if 0 in match['radiant_team'] or 0 in match['dire_team']:
        continue
    if len(match['radiant_team']) != 5 or len(match['dire_team']) != 5:
        continue

    winners = match['radiant_team'] if match['radiant_win'] else match['dire_team']
    losers  = match['dire_team'] if match['radiant_win'] else match['radiant_team']

    for i in range(5):
        for j in range(i+1, 5):
            win_pair  = tuple(sorted([winners[i], winners[j]]))
            lose_pair = tuple(sorted([losers[i], losers[j]]))

            if win_pair not in synergy:
                synergy[win_pair] = {'wins': 0, 'games': 0}
            if lose_pair not in synergy:
                synergy[lose_pair] = {'wins': 0, 'games': 0}

            synergy[win_pair]['wins']  += 1
            synergy[win_pair]['games'] += 1
            synergy[lose_pair]['games'] += 1

print(f"Built synergy for {len(synergy)} hero pairs")

# Synergy score helper function
def get_team_synergy(team, synergy):
    scores = []
    for i in range(5):
        for j in range(i+1, 5):
            pair = tuple(sorted([team[i], team[j]]))
            if pair in synergy and synergy[pair]['games'] > 0:
                score = synergy[pair]['wins'] / synergy[pair]['games']
            else:
                score = 0.5
            scores.append(score)
    return sum(scores) / len(scores)

# Fetch hero stats
fetcher         =   OpenDotaFetcher()
hero_winrates   =   fetcher.fetch_hero_winrates()

for match in matches:
    if 0 in match['radiant_team'] or 0 in match['dire_team']:
        continue

    if len(match['radiant_team']) != 5 or len(match['dire_team']) != 5:
        continue

    radiant_winrates =   [hero_winrates[hero_id]['win_rate'] for hero_id in match['radiant_team']]
    dire_winrates    =   [hero_winrates[hero_id]['win_rate'] for hero_id in match['dire_team']]
    radiant_synergy  =   get_team_synergy(match['radiant_team'], synergy)
    dire_synergy     =   get_team_synergy(match['dire_team'], synergy)
    features         =   radiant_winrates + dire_winrates + [radiant_synergy, dire_synergy]
    label            =   int(match['radiant_win'])
    X.append(features)
    y.append(label)

if len(X) == 0:
    print("No matches found in database. Run fetcher first.")
    exit(1)

print(f"Total matches {len(X)}")
print(f"Sample features {X[0]}")
print(f"Sample label {y[0]}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler  =   StandardScaler()
X_train =   scaler.fit_transform(X_train)
X_test  =   scaler.transform(X_test)

# Model Architecture
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(12,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Train
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save
model.save(f'{MODEL_DIR}/dota2_model.h5')
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
print("Model saved!")