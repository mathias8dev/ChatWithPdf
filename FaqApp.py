import os
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from models import db, init_db
from domain import app
import os
from utils import str2bool


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    is_debug = str2bool(os.environ.get("DEBUG"))
    app.run(debug=is_debug)
