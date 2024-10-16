import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

# Define parties and regions
parties = ['Labour', 'Conservative', 'Reform UK', 'Liberal Democrats', 'Green Party of England and Wales', 'Scottish National Party']
regions = ['Wales', 'Scotland', 'North East', 'North West', 'Yorkshire and The Humber', 'East Midlands', 
           'West Midlands', 'East of England', 'London', 'South East', 'South West']

# Adjusted distributions based on UK statistics
age_distribution = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
age_probabilities = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]  # Approximate age distribution
education_distribution = ['Degree', 'A-levels', 'GCSE', 'None']
education_probabilities = [0.3, 0.25, 0.25, 0.2]  # Approximate education distribution
income_distribution = ['0-£12,570', '£12,571-£50,270', '£50,271-£125,140', '£125,140+']
income_probabilities = [0.2, 0.5, 0.2, 0.1]  # Approximate income distribution

# Generate polling data with realistic distributions
polling_data = pd.DataFrame({
    'age': np.random.choice(age_distribution, size=1000, p=age_probabilities),
    'gender': np.random.choice(['Male', 'Female'], size=1000),
    'education': np.random.choice(education_distribution, size=1000, p=education_probabilities),
    'income': np.random.choice(income_distribution, size=1000, p=income_probabilities),
    'region': np.random.choice(regions, size=1000),
    'party_preference': np.random.choice(parties, size=1000)
})

# Generate constituency data with realistic population sizes
average_population_size = 75000  # Average constituency population
population_std_dev = 20000  # Standard deviation for population size
constituency_data = pd.DataFrame({
    'constituency': [f'Constituency {i+1}' for i in range(500)],
    'population': np.random.normal(loc=average_population_size, scale=population_std_dev, size=500).astype(int),
    'region': np.random.choice(regions, size=500)
})

# Clip populations to be at least 1000 (minimum constituency size)
constituency_data['population'] = constituency_data['population'].clip(lower=1000)

# Data Preparation
X = pd.get_dummies(polling_data[['age', 'gender', 'education', 'income', 'region']], drop_first=True)
y = polling_data['party_preference']

# Convert y to a categorical type
y = pd.Categorical(y)

# Convert boolean columns to integers
X = X.astype(int)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit a Multinomial Logistic Regression Model (Sklearn)
log_reg = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000)
log_reg.fit(X_train_scaled, y_train.codes)  # Use y_train.codes for categorical labels

# Define the full columns from the training data
full_columns = X_train.columns

# Function to process each constituency
def process_constituency(row, full_columns, scaler, log_reg, num_samples=100):
    constituency_votes = np.zeros(len(parties))  # Initialize votes array for the constituency

    for _ in range(num_samples):
        # Create demographic factors for the constituency
        demographic_data = {
            'age': random.choice(age_distribution),
            'gender': random.choice(['Male', 'Female']),
            'education': random.choice(education_distribution),
            'income': random.choice(income_distribution),
            'region': row['region']
        }

        # Create a DataFrame with a single row
        demographic_factors = pd.DataFrame(demographic_data, index=[0])

        # Create dummy variables
        demographic_factors = pd.get_dummies(demographic_factors, drop_first=True)

        # Ensure demographic_factors has all columns from the model
        demographic_factors = demographic_factors.reindex(columns=full_columns, fill_value=0)

        # Scale the features using the same scaler used on the training data
        demographic_factors_scaled = scaler.transform(demographic_factors)

        # Predict party preference probabilities for this demographic group
        predicted_probs = log_reg.predict_proba(demographic_factors_scaled)

        # Add the predicted probabilities to the total votes for this constituency
        constituency_votes += predicted_probs.flatten()

    # Multiply votes by the population size to scale to the actual constituency size
    constituency_votes *= row['population'] / num_samples
    return constituency_votes

# Apply parallel processing using joblib to speed up processing for each constituency
constituency_results = Parallel(n_jobs=-1)(
    delayed(process_constituency)(row, full_columns, scaler, log_reg, num_samples=100)
    for idx, row in tqdm(constituency_data.iterrows(), total=constituency_data.shape[0], desc="Processing Constituencies")
)

# Convert results into a 2D array (500 constituencies, 6 parties)
constituency_results_2d = np.array(constituency_results)

# Create a DataFrame for the results
constituency_results_df = pd.DataFrame(constituency_results_2d, columns=log_reg.classes_)

# Determine the winning party for each constituency
constituency_results_df['winner'] = constituency_results_df.idxmax(axis=1)  # Get the index of the winning party

# Combine with constituency names and region
final_results = pd.concat([constituency_data[['constituency', 'region']], constituency_results_df], axis=1)

# Convert winner index (which is numerical) into the actual party name
final_results['winner'] = final_results['winner'].map(lambda x: parties[x])  # Fix winner to show party names

# Count the number of seats won by each party overall
overall_results = final_results['winner'].value_counts()

# Count the number of seats won by each party in each region
regional_results = final_results.groupby('region')['winner'].value_counts().unstack(fill_value=0)

# Check the results
print("Overall seat count by party:\n", overall_results)
print("\nSeat count by party in each region:\n", regional_results)