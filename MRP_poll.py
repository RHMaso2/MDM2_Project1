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

# Define parties (assumed consistent with CSV)
parties = ['Labour', 'Conservative', 'Reform UK', 'Liberal Democrats', 'Green Party of England and Wales', 'Scottish National Party']

# Data Preparation
X = pd.get_dummies(polling_data[['age', 'gender', 'education', 'income', 'region']], drop_first=True)
y = pd.Categorical(polling_data['party_preference'])

# Ensure integer conversion for boolean columns
X = X.astype(int)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a Multinomial Logistic Regression Model
log_reg = LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter=1000)
log_reg.fit(X_train_scaled, y_train.codes)

# Define column structure from training data
full_columns = X_train.columns

# Function to simulate voting for each constituency
def process_constituency(row, full_columns, scaler, log_reg, num_samples=500):
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

        predicted_probs = log_reg.predict_proba(demographic_factors_scaled)
        sampled_vote = np.random.choice(len(parties), p=predicted_probs.flatten())
        constituency_votes[sampled_vote] += 1

    # Scale by population
    constituency_votes = (constituency_votes / num_samples) * row['population']
    return constituency_votes

# Parallel processing for each constituency
constituency_results = Parallel(n_jobs=-1)(
    delayed(process_constituency)(row, full_columns, scaler, log_reg)
    for idx, row in tqdm(constituency_data.iterrows(), total=constituency_data.shape[0], desc="Processing Constituencies")
)

# Convert results into DataFrame
constituency_results_df = pd.DataFrame(constituency_results, columns=log_reg.classes_)
constituency_results_df['winner'] = constituency_results_df.idxmax(axis=1)

# Combine results with constituency and region data
final_results = pd.concat([constituency_data[['constituency', 'region']], constituency_results_df], axis=1)

# Map index of winning party to party name
final_results['winner'] = final_results['winner'].map(lambda x: parties[x])

# Overall and regional results
overall_results = final_results['winner'].value_counts()
regional_results = final_results.groupby('region')['winner'].value_counts().unstack(fill_value=0)

# Display results
print("Overall seat count by party:\n", overall_results)
print("\nSeat count by party in each region:\n", regional_results)

# Plot overall seat count
overall_results.plot(kind='bar', title='Overall Seat Count by Party', ylabel='Number of Seats')
plt.show()

# Plot regional seat counts by party
regional_results.plot(kind='bar', stacked=True, title='Seats by Party in Each Region', ylabel='Number of Seats')
plt.show()
