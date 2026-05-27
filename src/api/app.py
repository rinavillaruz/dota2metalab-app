import os
import joblib
import numpy as np
from tensorflow import keras
from pymongo import MongoClient
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS

MONGO_URI   = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MODEL_DIR   = os.getenv('MODEL_DIR', 'models')
MODEL_PATH  = f"{MODEL_DIR}/dota2_model.h5"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"

HERO_NAMES = [
  {"id":1,"name":"Anti-Mage"},{"id":2,"name":"Axe"},{"id":3,"name":"Bane"},
  {"id":4,"name":"Bloodseeker"},{"id":5,"name":"Crystal Maiden"},
  {"id":6,"name":"Drow Ranger"},{"id":7,"name":"Earthshaker"},
  {"id":8,"name":"Juggernaut"},{"id":9,"name":"Mirana"},{"id":10,"name":"Morphling"},
  {"id":11,"name":"Shadow Fiend"},{"id":12,"name":"Phantom Lancer"},{"id":13,"name":"Puck"},
  {"id":14,"name":"Pudge"},{"id":15,"name":"Razor"},{"id":16,"name":"Sand King"},
  {"id":17,"name":"Storm Spirit"},{"id":18,"name":"Sven"},{"id":19,"name":"Tiny"},
  {"id":20,"name":"Vengeful Spirit"},{"id":21,"name":"Windranger"},{"id":22,"name":"Zeus"},
  {"id":23,"name":"Kunkka"},{"id":25,"name":"Lina"},{"id":26,"name":"Lion"},
  {"id":27,"name":"Shadow Shaman"},{"id":28,"name":"Slardar"},{"id":29,"name":"Tidehunter"},
  {"id":30,"name":"Witch Doctor"},{"id":31,"name":"Lich"},{"id":32,"name":"Riki"},
  {"id":33,"name":"Enigma"},{"id":34,"name":"Tinker"},{"id":35,"name":"Sniper"},
  {"id":36,"name":"Necrophos"},{"id":37,"name":"Warlock"},{"id":38,"name":"Beastmaster"},
  {"id":39,"name":"Queen of Pain"},{"id":40,"name":"Venomancer"},{"id":41,"name":"Faceless Void"},
  {"id":42,"name":"Wraith King"},{"id":43,"name":"Death Prophet"},{"id":44,"name":"Phantom Assassin"},
  {"id":45,"name":"Pugna"},{"id":46,"name":"Templar Assassin"},{"id":47,"name":"Viper"},
  {"id":48,"name":"Luna"},{"id":49,"name":"Dragon Knight"},{"id":50,"name":"Dazzle"},
  {"id":51,"name":"Clockwerk"},{"id":52,"name":"Leshrac"},{"id":53,"name":"Nature's Prophet"},
  {"id":54,"name":"Lifestealer"},{"id":55,"name":"Dark Seer"},{"id":56,"name":"Clinkz"},
  {"id":57,"name":"Omniknight"},{"id":58,"name":"Enchantress"},{"id":59,"name":"Huskar"},
  {"id":60,"name":"Night Stalker"},{"id":61,"name":"Broodmother"},{"id":62,"name":"Bounty Hunter"},
  {"id":63,"name":"Weaver"},{"id":64,"name":"Jakiro"},{"id":65,"name":"Batrider"},
  {"id":66,"name":"Chen"},{"id":67,"name":"Spectre"},{"id":68,"name":"Ancient Apparition"},
  {"id":69,"name":"Doom"},{"id":70,"name":"Ursa"},{"id":71,"name":"Spirit Breaker"},
  {"id":72,"name":"Gyrocopter"},{"id":73,"name":"Alchemist"},{"id":74,"name":"Invoker"},
  {"id":75,"name":"Silencer"},{"id":76,"name":"Outworld Destroyer"},{"id":77,"name":"Lycan"},
  {"id":78,"name":"Brewmaster"},{"id":79,"name":"Shadow Demon"},{"id":80,"name":"Lone Druid"},
  {"id":81,"name":"Chaos Knight"},{"id":82,"name":"Meepo"},{"id":83,"name":"Treant Protector"},
  {"id":84,"name":"Ogre Magi"},{"id":85,"name":"Undying"},{"id":86,"name":"Rubick"},
  {"id":87,"name":"Disruptor"},{"id":88,"name":"Nyx Assassin"},{"id":89,"name":"Naga Siren"},
  {"id":90,"name":"Keeper of the Light"},{"id":91,"name":"Io"},{"id":92,"name":"Visage"},
  {"id":93,"name":"Slark"},{"id":94,"name":"Medusa"},{"id":95,"name":"Troll Warlord"},
  {"id":96,"name":"Centaur Warrunner"},{"id":97,"name":"Magnus"},{"id":98,"name":"Timbersaw"},
  {"id":99,"name":"Bristleback"},{"id":100,"name":"Tusk"},{"id":101,"name":"Skywrath Mage"},
  {"id":102,"name":"Abaddon"},{"id":103,"name":"Elder Titan"},{"id":104,"name":"Legion Commander"},
  {"id":105,"name":"Techies"},{"id":106,"name":"Ember Spirit"},{"id":107,"name":"Earth Spirit"},
  {"id":108,"name":"Underlord"},{"id":109,"name":"Terrorblade"},{"id":110,"name":"Phoenix"},
  {"id":111,"name":"Oracle"},{"id":112,"name":"Winter Wyvern"},{"id":113,"name":"Arc Warden"},
  {"id":114,"name":"Monkey King"},{"id":119,"name":"Dark Willow"},{"id":120,"name":"Pangolier"},
  {"id":121,"name":"Grimstroke"},{"id":123,"name":"Hoodwink"},{"id":126,"name":"Void Spirit"},
  {"id":128,"name":"Snapfire"},{"id":129,"name":"Mars"},{"id":131,"name":"Ringmaster"},
  {"id":135,"name":"Dawnbreaker"},{"id":136,"name":"Marci"},{"id":137,"name":"Primal Beast"},
  {"id":138,"name":"Muerta"},{"id":145,"name":"Kez"},{"id":155,"name":"Largo"}
]

app = Flask(__name__)
CORS(app)

# Global state — populated by init_app_data() at startup, or overridden in tests
model        = None
scaler       = None
client       = None
db           = None
collection   = None
all_matches  = []
hero_winrates = {}
synergy      = {}


def init_app_data():
    """Load model, connect to MongoDB, and build hero stats + synergy matrix.
    Called once at startup. Skipped during tests so imports are fast.
    """
    global model, scaler, client, db, collection, all_matches, hero_winrates, synergy

    # Load model and scaler
    try:
        model  = keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model not loaded: {e}")
        model  = None
        scaler = None

    # MongoDB connection
    try:
        client     = MongoClient(MONGO_URI)
        db         = client['dota2metalab']
        collection = db['matches']
        client.admin.command('ping')
        print("MongoDB connected successfully")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        client     = None
        db         = None
        collection = None

    # Load all matches once — used for both win rates and synergy
    all_matches = list(collection.find({})) if collection else []
    print(f"Loaded {len(all_matches)} matches from MongoDB")

    # Calculate hero win rates from local matches
    print("Calculating hero win rates from local matches...")
    hero_stats = {}
    for match in all_matches:
        if 0 in match['radiant_team'] or 0 in match['dire_team']:
            continue
        if len(match['radiant_team']) != 5 or len(match['dire_team']) != 5:
            continue
        winners = match['radiant_team'] if match['radiant_win'] else match['dire_team']
        losers  = match['dire_team']    if match['radiant_win'] else match['radiant_team']
        for hero_id in winners:
            if hero_id not in hero_stats:
                hero_stats[hero_id] = {'wins': 0, 'games': 0}
            hero_stats[hero_id]['wins']  += 1
            hero_stats[hero_id]['games'] += 1
        for hero_id in losers:
            if hero_id not in hero_stats:
                hero_stats[hero_id] = {'wins': 0, 'games': 0}
            hero_stats[hero_id]['games'] += 1

    hero_winrates = {
        hid: {
            'win_rate':  s['wins'] / s['games'] if s['games'] > 0 else 0.5,
            'pick_rate': 0,
            'ban_rate':  0
        }
        for hid, s in hero_stats.items()
    }
    print(f"Calculated win rates for {len(hero_winrates)} heroes")

    # Build synergy matrix from local matches
    print("Building synergy matrix...")
    for match in all_matches:
        if 0 in match['radiant_team'] or 0 in match['dire_team']:
            continue
        if len(match['radiant_team']) != 5 or len(match['dire_team']) != 5:
            continue
        winners = match['radiant_team'] if match['radiant_win'] else match['dire_team']
        losers  = match['dire_team']    if match['radiant_win'] else match['radiant_team']
        for i in range(5):
            for j in range(i+1, 5):
                win_pair  = tuple(sorted([winners[i], winners[j]]))
                lose_pair = tuple(sorted([losers[i],  losers[j]]))
                if win_pair not in synergy:
                    synergy[win_pair] = {'wins': 0, 'games': 0}
                if lose_pair not in synergy:
                    synergy[lose_pair] = {'wins': 0, 'games': 0}
                synergy[win_pair]['wins']   += 1
                synergy[win_pair]['games']  += 1
                synergy[lose_pair]['games'] += 1
    print(f"Built synergy for {len(synergy)} hero pairs")


# Helper
def get_team_synergy(team, synergy):
    scores = []
    for i in range(5):
        for j in range(i+1, 5):
            pair = tuple(sorted([team[i], team[j]]))
            if pair in synergy and synergy[pair]['games'] > 0:
                scores.append(synergy[pair]['wins'] / synergy[pair]['games'])
            else:
                scores.append(0.5)
    return sum(scores) / len(scores)


@app.route('/health')
def health():
    return jsonify({
        'status':        'healthy',
        'model_loaded':  model is not None,
        'mongodb':       client is not None,
        'heroes_loaded': len(hero_winrates),
        'synergy_pairs': len(synergy)
    })


@app.route('/stats')
def stats():
    if collection is None:
        return jsonify({'error': 'MongoDB not connected'}), 503
    total        = collection.count_documents({})
    radiant_wins = collection.count_documents({'radiant_win': True})
    return jsonify({
        'total_matches': total,
        'radiant_wins':  radiant_wins,
        'dire_wins':     total - radiant_wins
    })


@app.route('/predict/draft', methods=['POST'])
def predict_draft():
    if model is None:
        return jsonify({'error': 'Model not loaded yet. Run trainer first.'}), 503
    data         = request.json
    radiant_team = data.get('radiant_team', [])
    dire_team    = data.get('dire_team', [])
    if len(radiant_team) != 5 or len(dire_team) != 5:
        return jsonify({'error': 'Each team must have exactly 5 heroes'}), 400
    if 0 in radiant_team or 0 in dire_team:
        return jsonify({'error': 'Invalid hero ID'}), 400
    for hero_id in radiant_team + dire_team:
        if hero_id not in hero_winrates:
            return jsonify({'error': f'Unknown hero ID: {hero_id}'}), 400
    radiant_winrates = [hero_winrates[hid]['win_rate'] for hid in radiant_team]
    dire_winrates    = [hero_winrates[hid]['win_rate'] for hid in dire_team]
    radiant_synergy  = get_team_synergy(radiant_team, synergy)
    dire_synergy     = get_team_synergy(dire_team, synergy)
    features         = radiant_winrates + dire_winrates + [radiant_synergy, dire_synergy]
    features_scaled  = scaler.transform(np.array([features]))
    prediction       = model.predict(features_scaled, verbose=0)[0][0]
    return jsonify({
        'radiant_win_probability': float(prediction),
        'dire_win_probability':    float(1 - prediction),
        'predicted_winner':        'Radiant' if prediction > 0.5 else 'Dire'
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    team = data.get('team', [])
    if len(team) == 0 or len(team) > 4:
        return jsonify({'error': 'Provide 1-4 heroes'}), 400
    picked     = set(team)
    candidates = []
    for h in HERO_NAMES:
        hero_id = h['id']
        if hero_id in picked:
            continue
        scores = []
        for ally_id in team:
            pair = tuple(sorted([hero_id, ally_id]))
            if pair in synergy and synergy[pair]['games'] > 0:
                scores.append(synergy[pair]['wins'] / synergy[pair]['games'])
            else:
                scores.append(0.5)
        avg_synergy = sum(scores) / len(scores)
        win_rate    = hero_winrates.get(hero_id, {}).get('win_rate', 0.5)
        combined    = (avg_synergy * 0.7) + (win_rate * 0.3)
        candidates.append({
            'hero_id':       hero_id,
            'hero_name':     h['name'],
            'synergy_score': avg_synergy,
            'win_rate':      win_rate,
            'combined':      combined
        })
    candidates.sort(key=lambda x: x['combined'], reverse=True)
    return jsonify({'recommendations': candidates[:5]})


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    init_app_data()
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)