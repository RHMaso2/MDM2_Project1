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

# Analyze the performance of the Conservative party in competitive constituencies.
# Iterate through each competitive constituency and compare the vote share of the Conservative party with other parties.
for index, row in competitive_constituencies.iterrows():
    con_share = row['Con_share']  # Conservative party vote share
    other_parties_shares = []  # List to store information on other parties with similar vote shares
    for party in parties:
        if party != 'Con':  # Compare with all other parties except Conservative
            party_share = row[f'{party}_share']
            # Select constituencies where the difference between Conservative and other party vote share is less than 5% (and both are greater than 0%)
            if abs(con_share - party_share) < 5 and con_share > 0 and party_share > 0:
                other_parties_shares.append(f" - {party} Vote Share: {party_share:.2f}% (close to Conservative)")

    # Only print constituencies that have significant comparisons (i.e., close competitors)
    if other_parties_shares:
        print(f"\nConstituency: {row['Constituency name']}")
        print(f"Conservative Vote Share: {con_share:.2f}%")
        for share_info in other_parties_shares:
            print(share_info)

# Create a boxplot to visualize the vote share distribution among major parties in competitive constituencies.
# This helps in understanding how the vote shares compare across different parties.
plt.figure(figsize=(12, 8))
sns.boxplot(data=competitive_constituencies[['Con_share', 'Lab_share', 'LD_share', 'RUK_share']])
plt.title('Vote Share Distribution by Party in Competitive Constituencies (2024)')
plt.xlabel('Party')
plt.ylabel('Vote Share (%)')
plt.tight_layout()
plt.show()

# Perform statistical tests (t-tests) to compare the Conservative party's vote share with other major parties.
# The goal is to see if there is a significant difference in vote shares between the Conservative party and others.
con_share = competitive_constituencies['Con_share'].dropna()
lab_share = competitive_constituencies['Lab_share'].dropna()
ld_share = competitive_constituencies['LD_share'].dropna()
ruk_share = competitive_constituencies['RUK_share'].dropna()

# Conduct t-tests between Conservative and Labour, Liberal Democrats, and RUK.
# A t-test is used to determine if there is a significant difference between the means of two groups.
t_stat_lab, p_value_lab = ttest_ind(con_share, lab_share, equal_var=False)
t_stat_ld, p_value_ld = ttest_ind(con_share, ld_share, equal_var=False)
t_stat_ruk, p_value_ruk = ttest_ind(con_share, ruk_share, equal_var=False)

# Print the t-test results for Conservative vs. other parties
print(f"\nT-test between Conservative and Labour: t-statistic = {t_stat_lab}, p-value = {p_value_lab}")
print(f"T-test between Conservative and Liberal Democrats: t-statistic = {t_stat_ld}, p-value = {p_value_ld}")
print(f"T-test between Conservative and RUK: t-statistic = {t_stat_ruk}, p-value = {p_value_ruk}")

# Analyze and interpret the t-test results.
# If p-value < 0.05, it means there is a statistically significant difference between the two parties' vote shares.
if p_value_lab < 0.05:
    print("There is a significant difference between Conservative and Labour vote shares in competitive constituencies.")
else:
    print("No significant difference between Conservative and Labour vote shares in competitive constituencies.")

if p_value_ld < 0.05:
    print("There is a significant difference between Conservative and Liberal Democrats vote shares in competitive constituencies.")
else:
    print("No significant difference between Conservative and Liberal Democrats vote shares in competitive constituencies.")

if p_value_ruk < 0.05:
    print("There is a significant difference between Conservative and RUK vote shares in competitive constituencies.")
else:
    print("No significant difference between Conservative and RUK vote shares in competitive constituencies.")
