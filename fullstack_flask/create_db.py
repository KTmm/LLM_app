#Run at the first time when starting the app
from src.app import app, db
with app.app_context():
    db.create_all()