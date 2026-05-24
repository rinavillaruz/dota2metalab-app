# 1. activate venv first
source venv/bin/activate

# 2. install
pip install -r requirements.txt

# 3. lock versions
pip freeze > requirements.txt

# 4. run the script
PYTHONPATH=. python scripts/fetch_data.py

# zip -r dota2metalab-app.zip .github .gitignore build docker-compose.yaml LICENSE README.md scripts src