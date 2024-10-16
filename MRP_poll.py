import random
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Load Data
polling_data = pd.read_csv('polling_data.csv')
constituency_data = pd.read_csv('constituency_data.csv')

# Step 2: Data Preparation
X = pd.get_dummies(polling_data[['age', 'gender', 'education', 'region']], drop_first=True)
y = polling_data['party_preference']

# Convert y to a categorical type
y = pd.Categorical(y)

# Convert boolean columns to integers
X = X.astype(int)

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X_train boolean columns to integers
X_train = X_train.astype(int)

# Remove constant columns
X_train = X_train.loc[:, (X_train != X_train.iloc[0]).any()]

# Step 4: Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit a Multinomial Logistic Regression Model (Sklearn)
log_reg = LogisticRegression( solver='lbfgs', penalty='l2', max_iter=1000 )
log_reg.fit(X_train_scaled, y_train.codes)  # Use y_train.codes for categorical labels

# Modify the predict_constituency function
def predict_constituency(constituency_row, model, full_columns, scaler):
    # Create demographic factors from the constituency row
    demographic_factors = pd.get_dummies(constituency_row[['age_group', 'gender', 'education_level']], drop_first=True)

    # Ensure demographic_factors has all columns from the model
    demographic_factors = demographic_factors.reindex(columns=full_columns, fill_value=0)

    # Scale the features using the same scaler used on the training data
    demographic_factors_scaled = scaler.transform(demographic_factors)

    # Predict party preference probabilities for this demographic group
    predicted_probs = model.predict_proba(demographic_factors_scaled)

    # Sum over all demographic groups (axis 0) to get total predicted votes
    votes = predicted_probs.sum(axis=0) * constituency_row['population']
    
    return votes

# Step 5: Apply the model to each constituency
constituency_results = []

# Define the full columns from the training data
full_columns = X_train.columns  # Ensure these are the columns used in training the model

for idx, row in constituency_data.iterrows():
    prediction = predict_constituency(row, log_reg, full_columns, scaler)
    constituency_results.append(prediction)

# Convert results into a 2D array (500 constituencies, 6 parties)
constituency_results_2d = np.array(constituency_results)

# Now that the results are 2D, create a DataFrame
constituency_results_df = pd.DataFrame(constituency_results_2d, columns=log_reg.classes_)

# Step 6: Determine the winning party for each constituency
constituency_results_df['winner'] = constituency_results_df.idxmax(axis=1)

# Combine with constituency names
final_results = pd.concat([constituency_data[['constituency']], constituency_results_df], axis=1)

# Step 7: Output the predicted winners in each constituency
print(final_results[['constituency', 'winner']])
