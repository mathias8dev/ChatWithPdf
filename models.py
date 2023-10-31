import uuid
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class DeviceModel(db.Model):
    __tablename__ = 'devices'

    id = db.Column(db.Integer, primary_key=True)
    device_name = db.Column(db.String(80))
    device_key = db.Column(db.String(80))

    def __init__(self, device_name):
        self.device_name = device_name
        self.device_key = f"app-{uuid.uuid4().hex}"

    def json(self):
        return {
            'device_name': self.device_name, 
            'device_key': self.device_key
        }

    @classmethod
    def find_by_name(cls, device_name):
        return cls.query.filter_by(device_name=device_name).first()

    @classmethod
    def find_by_device_key(cls, device_key):
        return cls.query.filter_by(device_key=device_key).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self):
        db.session.delete(self)
        db.session.commit()



def init_db():
    db.drop_all()
    db.create_all()
