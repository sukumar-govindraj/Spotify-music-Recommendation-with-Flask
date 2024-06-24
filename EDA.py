import pandas as pd
import statsmodels.api as sm

# Load the dataset
dataset_path = 'dataset_with_limited_artists.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(data.head())

# Get an overview of the dataset
print(data.info())


import pandas as pd

# Load the dataset
dataset_path = 'dataset_with_limited_artists.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(data.head())

# Get an overview of the dataset
print(data.info())

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot for track duration vs. popularity
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['duration_ms'], y=data['popularity'])
plt.title('Track Duration vs. Popularity')
plt.xlabel('Track Duration (ms)')
plt.ylabel('Popularity')
plt.show()


# Select only the numeric columns for correlation calculation
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
numeric_data = data[numeric_columns]

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

print(correlation_matrix)


# Create a new column for track duration in minutes
data['duration_min'] = data['duration_ms'] / 60000

# Create box plots for popularity across different track duration ranges
plt.figure(figsize=(12, 6))
sns.boxplot(x=pd.cut(data['duration_min'], bins=[0, 2, 4, 6, 8, 10, 12, 15]), y=data['popularity'])
plt.title('Popularity across Track Duration Ranges')
plt.xlabel('Track Duration (minutes)')
plt.ylabel('Popularity')
plt.show()

# Perform a simple linear regression to understand the relationship
X = data['duration_ms']
y = data['popularity']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())

# Plot the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['duration_ms'], y=data['popularity'], label='Data Points')
plt.plot(data['duration_ms'], model.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression: Track Duration vs. Popularity')
plt.xlabel('Track Duration (ms)')
plt.ylabel('Popularity')
plt.legend()
plt.show()
