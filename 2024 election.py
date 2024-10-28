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
bars = plt.bar(gallagher_indices.keys(), gallagher_indices.values(), color='skyblue')
plt.title('Gallagher Index by Election Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Gallagher Index', fontsize=14)
plt.xticks(list(gallagher_indices.keys()), fontsize=12)
plt.yticks(fontsize=12)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}',
             ha='center', va='bottom', fontsize=12, color='black')

#Show the plot
plt.show()

# Gallagher Index values for each year except 2024
gallagher_indices = [12.31, 10.74, 6.44, 10.25]

# Gallagher Index value for 2024
gallagher_2024 = 18.96

# Calculate mean and standard deviation
mean_gallagher = np.mean(gallagher_indices)
std_gallagher = np.std(gallagher_indices, ddof=1)

# Calculate 95% confidence interval
confidence_level = 0.95
degrees_freedom = len(gallagher_indices) - 1
t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

margin_of_error = t_critical * (std_gallagher / np.sqrt(len(gallagher_indices)))
confidence_interval = (mean_gallagher - margin_of_error, mean_gallagher + margin_of_error)

# Output the results
print(f"Mean Gallagher Index: {mean_gallagher:.2f}")
print(f"Standard Deviation: {std_gallagher:.2f}")
print(f"95% Confidence Interval: {confidence_interval}")

# Check if 2024 Gallagher Index is outside the confidence interval
if gallagher_2024 < confidence_interval[0] or gallagher_2024 > confidence_interval[1]:
    print(f"\nThe Gallagher Index for 2024 ({gallagher_2024}) is significantly different from previous elections (outside the 95% confidence interval).")
else:
    print(f"\nThe Gallagher Index for 2024 ({gallagher_2024}) is not significantly different from previous elections (within the 95% confidence interval).")



