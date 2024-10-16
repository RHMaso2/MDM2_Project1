import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define parties and regions
parties = [ 'Labour', 'Conservative', 'Reform UK', 'Liberal Democrats', 'Green Party of England and Wales',
 'Scottish National Party' ]

regions = [ 'Wales', 'Scotland', 'North East', 'North West', 'Yorkshire and The Humber', 'East Midlands', 
    'West Midlands', 'East of England', 'London', 'South East', 'South West']

# Generate polling data
polling_data = pd.DataFrame({
    'age': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], size=1000),
    'gender': np.random.choice(['Male', 'Female'], size=1000),
    'education': np.random.choice(['Degree', 'A-levels', 'GCSE', 'None'], size=1000),
    'income': np.random.choice(['0-£12,570', '£12,571-£50,270', '£50,271-£125,140', '£125,140+'], size=1000),
    'region': np.random.choice(regions, size=1000),
    'party_preference': np.random.choice(parties, size=1000) })

# Generate constituency data
constituency_data = pd.DataFrame({'constituency': [f'Constituency {i+1}' for i in range(500)],
    'population': np.random.randint(1000, 10000, size=500),'region': np.random.choice(regions, size=500)})

# Save to CSV
polling_data.to_csv('polling_data.csv', index=False)
constituency_data.to_csv('constituency_data.csv', index=False)

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

# Apply the model to each constituency
constituency_results = []

# Define the full columns from the training data
full_columns = X_train.columns  # Ensure these are the columns used in training the model

for idx, row in constituency_data.iterrows():
    # Create demographic factors for the constituency
    demographic_data = {
        'age': random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+']),
        'gender': random.choice(['Male', 'Female']),
        'education': random.choice(['Degree', 'A-levels', 'GCSE', 'None']),
        'income': random.choice(['0-£12,570', '£12,571-£50,270', '£50,271-£125,140', '£125,140+']),
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

    # Calculate predicted votes based on population size
    votes = predicted_probs * row['population']
    
    # Sum votes across parties for the constituency
    constituency_results.append(votes.flatten())

# Convert results into a 2D array (500 constituencies, 6 parties)
constituency_results_2d = np.array(constituency_results)

# Create a DataFrame
constituency_results_df = pd.DataFrame(constituency_results_2d, columns=log_reg.classes_)

# Determine the winning party for each constituency
constituency_results_df['winner'] = constituency_results_df.idxmax(axis=1)

# Combine with constituency names and region
final_results = pd.concat([constituency_data[['constituency', 'region']], constituency_results_df], axis=1)

# Count the number of seats won by each party overall
overall_results = final_results['winner'].value_counts()

# Count the number of seats won by each party in each region
regional_results = final_results.groupby('region')['winner'].value_counts().unstack(fill_value=0)

# Check the results
print("Overall seat count by party:\n", overall_results)
print("\nSeat count by party in each region:\n", regional_results)

# Plot pie charts for overall results and regional results
def plot_pie_chart(data, title):
    plt.figure(figsize=(8, 6))
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    plt.show()

# Overall results pie chart
plot_pie_chart(overall_results, "Overall Seat Count by Party")

# Regional results pie charts
#for region in regional_results.index:
#    plot_pie_chart(regional_results.loc[region], f"Seat Count by Party in {region}")
