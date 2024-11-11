import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load the CSV file containing the election data
data = pd.read_csv('2024.csv')

# Convert the columns 'Electorate', 'Valid votes', and 'Majority' to numeric values for analysis.
# Any non-numeric values will be set to NaN (using 'coerce').
data['Electorate'] = pd.to_numeric(data['Electorate'], errors='coerce')
data['Valid votes'] = pd.to_numeric(data['Valid votes'], errors='coerce')
data['Majority'] = pd.to_numeric(data['Majority'], errors='coerce')

# Convert vote counts for each party to numeric values.
# Parties include 'Con' (Conservative), 'Lab' (Labour), 'LD' (Liberal Democrats), 'RUK', 'Green', and 'SNP'.
# Again, any non-numeric values will be set to NaN.
parties = ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP']
for party in parties:
    data[party] = pd.to_numeric(data[party], errors='coerce')

# Calculate the vote share percentage for each party.
# Vote share is computed as the number of votes received by the party divided by the total valid votes in the constituency, multiplied by 100 to get a percentage.
for party in parties:
    data[f'{party}_share'] = (data[party] / data['Valid votes']) * 100

# Calculate the percentage of difference (Majority / Electorate) for each constituency.
# Select constituencies where the difference is less than 5%, indicating competitive constituencies.
data['difference_percentage'] = (data['Majority'] / data['Electorate']) * 100
competitive_constituencies = data[data['difference_percentage'] < 5]

# Filter out constituencies where all parties have zero vote shares.
# We only want to analyze constituencies with at least one party having received votes.
competitive_constituencies = competitive_constituencies[
    (competitive_constituencies[parties].sum(axis=1) > 0)
]

# Create heatmaps to visualize the vote share distribution among major parties in competitive constituencies.
# Split the constituencies into parts to make the visualization clearer.
num_constituencies = len(competitive_constituencies)
num_parts = 4  # Split into 4 parts for better visualization
constituencies_per_part = num_constituencies // num_parts

for i in range(num_parts):
    start_idx = i * constituencies_per_part
    if i == num_parts - 1:
        end_idx = num_constituencies  # Include any remaining constituencies in the last part
    else:
        end_idx = (i + 1) * constituencies_per_part

    subset = competitive_constituencies.iloc[start_idx:end_idx]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        subset[[f'{party}_share' for party in parties]],
        annot=False,  # Disable annotation to reduce clutter
        cmap='Reds',
        cbar=True,
        linewidths=.4
    )
    plt.title(f'Heatmap of Vote Shares by Party in Competitive Constituencies (Image {i + 1})', fontsize=18)
    plt.xlabel('Party', fontsize=16)
    plt.ylabel('Constituency Index', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
