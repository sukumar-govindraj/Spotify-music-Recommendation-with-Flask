import pandas as pd

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Drop rows with NaN values
dataset.dropna(inplace=True)

# Check for duplicates in the 'track_id' column
duplicates = dataset[dataset.duplicated('track_id')]
print(f"Number of duplicate track IDs: {len(duplicates)}")

# Drop duplicates
dataset = dataset.drop_duplicates('track_id')

# Save the cleaned dataset
dataset.to_csv('dataset_cleaned.csv', index=False)
