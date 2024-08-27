# Spotify Music Recommendation System

## Introduction

The Spotify Music Recommendation System is a Flask-based web application that suggests music tracks based on the user's input track name. The recommendation model is built using TensorFlow, and the data is managed using SQLAlchemy with a SQL Server backend.

This project involves several components:
- Data Loading and Cleaning
- Exploratory Data Analysis (EDA)
- Model Training and Evaluation
- Flask-based Web Interface for User Interaction

## Features

- **User Input**: Enter a track name to receive music recommendations.
- **Music Recommendation Model**: Suggests songs similar to the input track using machine learning techniques.
- **Web Interface**: Simple and user-friendly interface to interact with the recommendation system.

## Setup and Installation

To set up and run the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/spotify-music-recommendation-system.git
    cd spotify-music-recommendation-system
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    The required packages include:
    - Flask
    - Flask-SQLAlchemy
    - pandas
    - tensorflow
    - scikit-learn
    - pyodbc (for SQL Server connectivity)

3. Set up your SQL Server database:
    - Create a new database named `MusicRecommendationDB`.
    - Update the SQL Server configuration in `app.py` to match your environment.
    - Run the database migration script if applicable or manually insert your data.

4. Run the Flask application:

    ```bash
    python app.py
    ```

5. Open your browser and go to `http://127.0.0.1:5000/` to access the application.

## Data and Preprocessing

The dataset used for this project contains information about various songs, including track names, artist names, album names, and other features. Data preprocessing is performed using the following scripts:
- **`data_clean.py`**: Cleans and preprocesses the dataset.
- **`load_data.py`**: Loads the cleaned dataset into the database.
- **`create_tables.py`**: Sets up the necessary database tables.

The final dataset is stored in `dataset_with_limited_artists.csv`.

## Exploratory Data Analysis (EDA)

Before building the recommendation model, we conducted an Exploratory Data Analysis (EDA) to understand the distribution of data and important features. The EDA script is available in `EDA.py`. Key insights include:
- Distribution of genres, artists, and albums.
- Correlation between features like danceability, energy, and popularity.
- Visualization of song features using histograms and scatter plots.

**Example visualizations:**
- Genre Distribution:
  ![Genre Distribution](images/genre_distribution.png)
- Feature Correlation:
  ![Feature Correlation](images/feature_correlation.png)

## Model Training

The recommendation model is built using a neural network implemented in TensorFlow. The model is trained on the cleaned dataset and saved as `recommendation_model.h5`. The preprocessing steps are saved using `joblib` in `preprocessor.joblib`.

The model training and evaluation are managed through the following script:
- **`models.py`**: Contains the code for training the recommendation model.

## Web Interface

The user interface is built using Flask and simple HTML. Users can input a track name to receive song recommendations. The HTML file for the interface is `index.html`, and the main Flask app is in `app.py`.

### UI Screenshots

**Home Page:**
![image](https://github.com/user-attachments/assets/56805287-1984-42a6-a715-14588895ac67)



**Recommendation Results:**
![image](https://github.com/user-attachments/assets/13e7de7b-0ad8-496c-97d2-8bf5bd1abd2c)

## How to Use

1. Enter the track name in the input box on the home page.
2. Click on "Get Recommendations."
3. View the list of recommended songs based on the input track.

