from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import urllib
from models import db

app = Flask(__name__)

# SQL Server configuration using Windows Authentication
params = urllib.parse.quote_plus(r"DRIVER={ODBC Driver 17 for SQL Server};"
                                r"SERVER=SUKUMAR\SQLEXPRESS;"  # Use raw string for server name
                                r"DATABASE=MusicRecommendationDB;"  # Use your created database name
                                r"Trusted_Connection=yes;")
app.config['SQLALCHEMY_DATABASE_URI'] = f'mssql+pyodbc:///?odbc_connect={params}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()  # This will create all the tables defined in your models
    print("All tables created successfully.")
