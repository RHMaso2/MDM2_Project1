import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the Mean Squared Error
def calc_mse(data):
    # Convert to percentages
    votes = data['Vote share']  # Already in percentage
    seats = (data['Seats'] / 600) * 100  # Convert seat count to percentage
    
    # Calculate Mean Squared Error
    mse = (votes - seats) ** 2
    return mse

# Function to calculate the Gallagher Index
def G_index(mse):
    # Root Mean Square (RMS) and apply 0.5 factor
    gall_index = 0.5 * np.sqrt(np.sum(mse))
    return gall_index

# Load the election data for each year
data2024 = pd.read_csv('2024 data.csv')
data2019 = pd.read_csv('2019 data.csv')
data2017 = pd.read_csv('2017 data.csv')
data2015 = pd.read_csv('2015 data.csv')
data2010 = pd.read_csv('2010 data.csv')

# Create a dictionary to store the results
gallagher_indices = {}

# List of years and corresponding datasets
years = [2010,2015,2017,2019,2024]
datasets = [data2010,data2015,data2017,data2019,data2024]

# Calculate Gallagher Index for each year
for year, data in zip(years, datasets):
    mse = calc_mse(data)
    gallagher_indices[str(year)] = G_index(mse)



# Plotting the Gallagher Index for each year
plt.figure(figsize=(10, 6))
plt.bar(gallagher_indices.keys(), gallagher_indices.values(), color='skyblue')
plt.title('Gallagher Index by Election Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Gallagher Index', fontsize=14)
plt.xticks(list(gallagher_indices.keys()), fontsize=12)
plt.yticks(fontsize=12)
plt.show()