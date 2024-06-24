from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import urllib
from models import db, Track

app = Flask(__name__)

# SQL Server configuration using Windows Authentication
params = urllib.parse.quote_plus(r"DRIVER={ODBC Driver 17 for SQL Server};"
                                r"SERVER=SUKUMAR\SQLEXPRESS;"  # Use raw string for server name
                                r"DATABASE=MusicRecommendationDB;"  # Use your created database name
                                r"Trusted_Connection=yes;")
app.config['SQLALCHEMY_DATABASE_URI'] = f'mssql+pyodbc:///?odbc_connect={params}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def load_data_to_db():
    dataset = pd.read_csv('dataset_with_limited_artists.csv')  # Use the new dataset
    with app.app_context():
        for index, row in dataset.iterrows():
            # Ensure correct data types and handle potential issues
            track_id = str(row['track_id'])
            artist1 = str(row['artist1']) if 'artist1' in row and not pd.isna(row['artist1']) else None
            artist2 = str(row['artist2']) if 'artist2' in row and not pd.isna(row['artist2']) else None
            artist3 = str(row['artist3']) if 'artist3' in row and not pd.isna(row['artist3']) else None
            artist4 = str(row['artist4']) if 'artist4' in row and not pd.isna(row['artist4']) else None
            artist5 = str(row['artist5']) if 'artist5' in row and not pd.isna(row['artist5']) else None
            album_name = str(row['album_name'])
            track_name = str(row['track_name'])
            popularity = int(row['popularity'])
            duration_ms = int(row['duration_ms'])
            explicit = bool(row['explicit'])
            danceability = float(row['danceability'])
            energy = float(row['energy'])
            loudness = float(row['loudness'])
            mode = int(row['mode'])
            speechiness = float(row['speechiness'])
            acousticness = float(row['acousticness'])
            instrumentalness = float(row['instrumentalness'])
            liveness = float(row['liveness'])
            valence = float(row['valence'])
            tempo = float(row['tempo'])
            time_signature = int(row['time_signature'])
            track_genre = str(row['track_genre'])

            if not Track.query.filter_by(track_id=track_id).first():  # Check if entry already exists
                track = Track(
                    track_id=track_id,
                    artist1=artist1,
                    artist2=artist2,
                    artist3=artist3,
                    artist4=artist4,
                    artist5=artist5,
                    album_name=album_name,
                    track_name=track_name,
                    popularity=popularity,
                    duration_ms=duration_ms,
                    explicit=explicit,
                    danceability=danceability,
                    energy=energy,
                    loudness=loudness,
                    mode=mode,
                    speechiness=speechiness,
                    acousticness=acousticness,
                    instrumentalness=instrumentalness,
                    liveness=liveness,
                    valence=valence,
                    tempo=tempo,
                    time_signature=time_signature,
                    track_genre=track_genre
                )
                db.session.add(track)
        db.session.commit()

# Uncomment the following line to load the data into the database
load_data_to_db()
