from endpoints import app
from models import db, init_db
import os
from utils import generate_and_save_access_token
# Connect sqlalchemy to app
db.init_app(app)
# Check if the database was already initiated. If not, initialize it.
db_path = os.path.join("storage", "app.db")
if (not os.path.exists(db_path)):
    with app.app_context():
        init_db()

# Generate apiKey and save it
if (not os.path.exists("apiKey.txt")):
    generate_and_save_access_token()


