from endpoints import app
from models import db, init_db
import os

# Connect sqlalchemy to app
db.init_app(app)
# Check if the database was already initiated. If not, initialize it.
db_path = os.path.join("storage", "app.db")
if (not os.path.exists(db_path)):
    with app.app_context():
        init_db()


