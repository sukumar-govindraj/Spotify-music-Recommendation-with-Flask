import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib

# Load and preprocess data
dataset = pd.read_csv('dataset_with_limited_artists.csv')

# Select features including artist1 and track_genre
features = dataset[['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'artist1', 'track_genre']]
target = dataset['popularity']

# Define the column transformer to handle both numeric and categorical data
numeric_features = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
categorical_features = ['artist1', 'track_genre']

# Create the column transformer with one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Preprocess the features to determine the input shape for the model
preprocessed_features = preprocessor.fit_transform(features)
input_shape = preprocessed_features.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_features, target, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Input(shape=(input_shape,)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the model and the preprocessing pipeline
model.save('recommendation_model.h5')
joblib.dump(preprocessor, 'preprocessor.joblib')
