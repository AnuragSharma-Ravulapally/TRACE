# models.py
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()

class User(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.Text, nullable=False)  # JSON string of 512 floats

    def set_embedding(self, embedding_list):
        self.embedding = json.dumps(embedding_list)

    def get_embedding(self):
        return json.loads(self.embedding)