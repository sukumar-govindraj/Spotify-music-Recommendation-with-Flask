from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
import urllib
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
from models import db, Track

app = Flask(__name__)

# SQL Server configuration using Windows Authentication
params = urllib.parse.quote_plus(r"DRIVER={ODBC Driver 17 for SQL Server};"
                                r"SERVER=SUKUMAR\SQLEXPRESS;"
                                r"DATABASE=MusicRecommendationDB;"
                                r"Trusted_Connection=yes;")
app.config['SQLALCHEMY_DATABASE_URI'] = f'mssql+pyodbc:///?odbc_connect={params}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)  # Initialize SQLAlchemy with the Flask app

# Load the model and preprocessor
model = tf.keras.models.load_model('recommendation_model.h5')
preprocessor = joblib.load('preprocessor.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_data = request.json
        track_name = user_data.get('track_name')
        artist_name = user_data.get('artist_name', None)  # Get the artist name if provided

        # Debugging: Log the received data
        print(f"Received track_name: {track_name}, artist_name: {artist_name}")

        # Modify the query to include both track name and artist name if provided
        query = Track.query.filter(Track.track_name.ilike(f"%{track_name}%"))
        if artist_name:
            query = query.filter(Track.artist1.ilike(f"%{artist_name}%"))

        track = query.first()
        
        if not track:
            print("Track not found")
            return jsonify({'error': 'Track not found'}), 404

        # Extract features for the provided track
        features = {
            'danceability': track.danceability,
            'energy': track.energy,
            'loudness': track.loudness,
            'mode': track.mode,
            'speechiness': track.speechiness,
            'acousticness': track.acousticness,
            'instrumentalness': track.instrumentalness,
            'liveness': track.liveness,
            'valence': track.valence,
            'tempo': track.tempo,
            'time_signature': track.time_signature,
            'artist1': track.artist1,
            'track_genre': track.track_genre
        }

        features_df = pd.DataFrame([features])

        # Preprocess the features using the same preprocessor used during training
        normalized_features = preprocessor.transform(features_df)

        # Predict using the model
        prediction = model.predict(normalized_features)[0]

        # Convert the prediction to a native Python float
        predicted_popularity = float(prediction[0])

        # Retrieve all tracks from the database for similarity comparison
        all_tracks = Track.query.all()
        all_features_df = pd.DataFrame([{
            'danceability': t.danceability,
            'energy': t.energy,
            'loudness': t.loudness,
            'mode': t.mode,
            'speechiness': t.speechiness,
            'acousticness': t.acousticness,
            'instrumentalness': t.instrumentalness,
            'liveness': t.liveness,
            'valence': t.valence,
            'tempo': t.tempo,
            'time_signature': t.time_signature,
            'artist1': t.artist1,
            'track_genre': t.track_genre
        } for t in all_tracks])

        normalized_all_features = preprocessor.transform(all_features_df)

        # Compute cosine similarity between the provided track and all other tracks
        similarities = cosine_similarity(normalized_features, normalized_all_features)[0]

        # Get the indices of the top 10 most similar tracks
        top_indices = similarities.argsort()[-10:][::-1]

        # Enforce diversity in the recommendations
        recommended_tracks = []
        seen_tracks = set()

        for i in top_indices:
            recommended_track = all_tracks[i]
            if recommended_track.track_name not in seen_tracks:
                recommended_tracks.append(recommended_track)
                seen_tracks.add(recommended_track.track_name)

            if len(recommended_tracks) >= 10:
                break

        recommendations = [{'track_name': track.track_name, 'artists': ';'.join(filter(None, [track.artist1, track.artist2, track.artist3, track.artist4, track.artist5])), 'album_name': track.album_name} for track in recommended_tracks]

        # Debugging: Log the recommendations
        print(f"Recommendations: {recommendations}")

        return jsonify({'recommendations': recommendations})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/check-data')
def check_data():
    tracks = Track.query.limit(10).all()
    data = [
        {
            'track_id': track.track_id,
            'track_name': track.track_name,
            'artists': ';'.join(filter(None, [track.artist1, track.artist2, track.artist3, track.artist4, track.artist5])),
            'album_name': track.album_name
        }
        for track in tracks
    ]
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
