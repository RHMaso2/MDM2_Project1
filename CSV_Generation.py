import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

# Define parties and regions
parties = ['Labour', 'Conservatives', 'Reform UK', 'Lib-Dems', 'Greens', 'SNP']
regions = {
    'Wales': 32,
    'Scotland': 57,
    'North East': 27,
    'North West': 73,
    'Yorkshire and The Humber': 54,
    'East Midlands': 47,
    'West Midlands': 57,
    'East of England': 61,
    'London': 75,
    'South East': 91,
    'South West': 58
}

# Adjusted distributions based on UK statistics
age_distribution = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
age_probabilities = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]  # Approximate age distribution
education_distribution = ['Degree', 'A-levels', 'GCSE', 'None']
education_probabilities = [0.3, 0.25, 0.25, 0.2]  # Approximate education distribution
income_distribution = ['0-£12,570', '£12,571-£50,270', '£50,271-£125,140', '£125,140+']
income_probabilities = [0.2, 0.5, 0.2, 0.1]  # Approximate income distribution

# Define regional party preference tendencies for the 2024 election
party_preferences_by_region = {
    'Wales': [0.35, 0.25, 0.05, 0.15, 0.1, 0.1],
    'Scotland': [0.2, 0.2, 0.05, 0.1, 0.05, 0.4],
    'North East': [0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
    'North West': [0.35, 0.3, 0.1, 0.15, 0.05, 0.05],
    'Yorkshire and The Humber': [0.35, 0.3, 0.1, 0.1, 0.1, 0.05],
    'East Midlands': [0.3, 0.35, 0.1, 0.1, 0.1, 0.05],
    'West Midlands': [0.3, 0.35, 0.1, 0.1, 0.1, 0.05],
    'East of England': [0.25, 0.4, 0.1, 0.15, 0.05, 0.05],
    'London': [0.4, 0.2, 0.05, 0.2, 0.1, 0.05],
    'South East': [0.25, 0.4, 0.1, 0.15, 0.05, 0.05],
    'South West': [0.25, 0.4, 0.1, 0.15, 0.05, 0.05]
}

# Generate polling data with realistic distributions and election-based party preference
polling_data = pd.DataFrame({
    'age': np.random.choice(age_distribution, size=1000, p=age_probabilities),
    'gender': np.random.choice(['Male', 'Female'], size=1000),
    'education': np.random.choice(education_distribution, size=1000, p=education_probabilities),
    'income': np.random.choice(income_distribution, size=1000, p=income_probabilities),
    'region': np.random.choice(list(regions.keys()), size=1000)
})

# Assign party preferences based on region-specific probabilities
polling_data['party_preference'] = polling_data['region'].apply(
    lambda region: np.random.choice(parties, p=party_preferences_by_region[region])
)

# Generate constituency data by region
constituency_list = []
average_population_size = 75000  # Average constituency population
population_std_dev = 20000  # Standard deviation for population size

for region, count in regions.items():
    for i in range(count):
        population = max(1000, int(np.random.normal(average_population_size, population_std_dev)))
        constituency_list.append({'constituency': f'{region} Constituency {i+1}', 'population': population, 'region': region})

constituency_data = pd.DataFrame(constituency_list)

# Data Preparation for the predictive model (if required)
X = pd.get_dummies(polling_data[['age', 'gender', 'education', 'income', 'region']], drop_first=True)
y = polling_data['party_preference']

# Export data to CSV files
polling_data.to_csv('polling_data.csv', index=False)
constituency_data.to_csv('constituency_data.csv', index=False)
