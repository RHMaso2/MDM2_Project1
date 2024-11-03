import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('HoC-GE2024-results-by-constituency(1).csv')

FPP_votes = df["First party"]

FPP_results = [0,0,0,0,0,0]

for i in FPP_votes:
    if i == "Con":
        FPP_results[0] += 1
    elif i == "Lab":
        FPP_results[1] += 1
    elif i == "LD":
        FPP_results[2] += 1
    elif i == "RUK":
        FPP_results[3] += 1
    elif i == "Green":
        FPP_results[4] += 1
    elif i == "SNP":
        FPP_results[5] += 1

party_votes = df.iloc[:, 16:22].replace(',', '', regex=True).to_numpy().astype(int)
party_votes_tot = []
for i in range(6):
    party_votes_tot.append(sum(party_votes[:, i]))

print(party_votes_tot)
print(FPP_results)

#---------------------------------------------------------

df2 = pd.read_csv('HoC-GE2019-results-by-constituency.csv')

FPP_votes2 = df2["First party"]

FPP_results2 = [0,0,0,0,0,0]

for i in FPP_votes2:
    if i == "Con":
        FPP_results2[0] += 1
    elif i == "Lab":
        FPP_results2[1] += 1
    elif i == "LD":
        FPP_results2[2] += 1
    elif i == "BRX":
        FPP_results2[3] += 1
    elif i == "Green":
        FPP_results2[4] += 1
    elif i == "SNP":
        FPP_results2[5] += 1

party_votes2 = df2.iloc[:, 18:24].replace(',', '', regex=True).to_numpy().astype(int)
party_votes_tot2 = []
for i in range(6):
    party_votes_tot2.append(sum(party_votes2[:, i]))

print(party_votes_tot2)
print(FPP_results2)

#---------------------------------------------------------

df3 = pd.read_csv('HoC-GE2015-results-by-constituency.csv')

FPP_votes3 = df3["First party"]

FPP_results3 = [0,0,0,0,0,0]

for i in FPP_votes3:
    if i == "Con":
        FPP_results3[0] += 1
    elif i == "Lab":
        FPP_results3[1] += 1
    elif i == "LD":
        FPP_results3[2] += 1
    elif i == "UKIP":
        FPP_results3[3] += 1
    elif i == "Green":
        FPP_results3[4] += 1
    elif i == "SNP":
        FPP_results3[5] += 1

party_votes3 = df3.iloc[:, 18:24].replace(',', '', regex=True).to_numpy().astype(int)
party_votes_tot3 = []
for i in range(6):
    party_votes_tot3.append(sum(party_votes3[:, i]))

print(party_votes_tot3)
print(FPP_results3)

import matplotlib.pyplot as plt
import pandas as pd

# Assuming party_votes_tot2, FPP_results2, party_votes_tot3, and FPP_results3 are already defined
# Example data for demonstration
years = ["2024", "2019", "2015"]
parties = ["Conservative", "Labour", "LibDem", "Reform/Brexit", "Green", "SNP"]
colors = ["blue", "red", "yellow", "darkred", "green", "purple"]

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs = axs.flatten()

# Data for each year
data = [
    (party_votes_tot, FPP_results),
    (party_votes_tot2, FPP_results2),
    (party_votes_tot3, FPP_results3)
]

for i, (votes, seats) in enumerate(data):
    for j, party in enumerate(parties):
        axs[i].scatter(votes[j], seats[j], color=colors[j], label=party)
    axs[i].set_title(f'Votes vs Seats in {years[i]}')
    axs[i].set_xlabel('Total Votes (ten thousands)')
    axs[i].set_ylabel('Seats Won')
    axs[i].legend()

# Adjust layout
plt.tight_layout()
plt.show()
