import numpy as np
import pandas as pd

df = pd.read_csv('HoC-GE2024-results-by-constituency(1).csv')

SEATS = 975
LIST_SEATS_TOTAL = 325

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

tot_party_votes = sum(party_votes_tot)
scale_factor = SEATS/tot_party_votes

proportion = []

for i in party_votes_tot:
    temp = i * scale_factor
    proportion.append(temp/SEATS)

list_seats = []
added_seats = []


# Calculate entitled seats
entitled_seats = [p * SEATS for p in proportion]
print("Entitled Seats:", entitled_seats)

print("Party Votes Total:", party_votes_tot)
print("Proportion:", proportion)
print("Entitled Seats:", entitled_seats)
print("FPP Results:", FPP_results)

# Initial allocation of seats using FPP results
initial_seats = FPP_results[:]
remaining_seats = SEATS - sum(FPP_results)
divisors = [1] * len(entitled_seats)

# Adjust entitled seats by subtracting FPP results
adjusted_entitled_seats = [entitled_seats[i] - FPP_results[i] for i in range(len(entitled_seats))]

# Distribute remaining seats using Sainte-Laguë method
for _ in range(remaining_seats):
    quotients = [adjusted_entitled_seats[i] / divisors[i] for i in range(len(adjusted_entitled_seats))]
    max_index = quotients.index(max(quotients))
    initial_seats[max_index] += 1
    divisors[max_index] += 2

# Check for overhang seats
total_seats = sum(initial_seats)
if total_seats > SEATS:
    print("Overhang seats detected. Total seats:", total_seats)
else:
    print("Final Seats:", initial_seats)