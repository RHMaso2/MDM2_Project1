import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Load CSV files
polling_data = pd.read_csv('polling_data.csv')
constituency_data = pd.read_csv('constituency_data.csv')

# Define parties and regions
parties = ['Labour', 'Conservative', 'Reform UK', 'Lib-Dems', 'Greens', 'SNP']
regions = constituency_data['region'].unique()

# Data Preparation
# Convert categorical data to numeric codes if not already in numeric form
X = pd.get_dummies(polling_data[['age', 'gender', 'education', 'income', 'region']], drop_first=True)
y = pd.Categorical(polling_data['party_preference']).codes

# Split data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a Multinomial Logistic Regression Model                                                                
log_reg = LogisticRegression(solver='saga', penalty='l2', C=1.5, max_iter=1000, random_state=42, class_weight={0:0.6, 1:2, 2:0.8, 3:1, 4:0.4, 5:0.4})
log_reg.fit(X_train_scaled, y_train)

# Define column structure for dummy variables
full_columns = X_train.columns

# Function to simulate voting for each constituency
def process_constituency(row, full_columns, scaler, log_reg, num_samples=100):
    constituency_votes = np.zeros(len(parties))  # Vote counts for each party

    for _ in range(num_samples):
        demographic_data = {
            'age': random.choice(polling_data['age'].unique()),
            'gender': random.choice(polling_data['gender'].unique()),
            'education': random.choice(polling_data['education'].unique()),
            'income': random.choice(polling_data['income'].unique()),
            'region': row['region']
        }
        
        demographic_factors = pd.DataFrame(demographic_data, index=[0])
        demographic_factors = pd.get_dummies(demographic_factors, drop_first=True)
        demographic_factors = demographic_factors.reindex(columns=full_columns, fill_value=0)
        demographic_factors_scaled = scaler.transform(demographic_factors)

        # Predict and sample vote based on predicted probabilities
        predicted_probs = log_reg.predict_proba(demographic_factors_scaled).flatten()
        sampled_vote = np.random.choice(len(parties), p=predicted_probs)
        constituency_votes[sampled_vote] += 1

    # Scale by population
    constituency_votes = (constituency_votes / num_samples) * row['population']
    return constituency_votes

# Process each constituency in parallel
constituency_results = Parallel(n_jobs=-1)(
    delayed(process_constituency)(row, full_columns, scaler, log_reg)
    for idx, row in tqdm(constituency_data.iterrows(), total=constituency_data.shape[0])
)

# Convert results into DataFrame
constituency_results_df = pd.DataFrame(constituency_results, columns=parties)
# Get the index of the party with the highest vote count
constituency_results_df['winner'] = constituency_results_df.values.argmax(axis=1)

# Combine results with constituency data
final_results = pd.concat([constituency_data[['constituency', 'region']], constituency_results_df], axis=1)

# Map the index of the winning party to the party name
final_results['winner'] = final_results['winner'].map(lambda x: parties[x])

# Overall and regional results
overall_results = final_results['winner'].value_counts()
regional_results = final_results.groupby('region')['winner'].value_counts().unstack(fill_value=0)

# Display results
print("Overall seat count by party:\n", overall_results)
print("\nSeat count by party in each region:\n", regional_results)

# Plot overall seat count
plt.figure(figsize=(10, 6))
overall_results.plot(kind='bar', title='Overall Seat Count by Party', ylabel='Number of Seats')
plt.show()

# Plot regional seat counts by party
plt.figure(figsize=(12, 8))
regional_results.plot(kind='bar', stacked=True, title='Seats by Party in Each Region', ylabel='Number of Seats')
plt.tight_layout()
plt.show()


