import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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

entitled_seats = []
for i in proportion:
    entitled_seats.append(math.ceil(i * SEATS))

for i in range(6):
    # list seats
    if entitled_seats[i] > FPP_results[i]:
        list_seats.append((entitled_seats[i]) - FPP_results[i])
        added_seats.append(0)
    else:
        # overhang seats
        added_seats.append(FPP_results[i] - entitled_seats[i])


print(party_votes_tot)
print(proportion)
print(added_seats)
print(list_seats)   # Seats to be added to FPP results to reach the entitled seats
print(entitled_seats)
# print(FPP_results)

difference = []

final_seats = FPP_results.copy()

# Initial allocation of seats
initial_seats = [int(entitled_seats[i]) for i in range(len(entitled_seats))]
remainders = [entitled_seats[i] - initial_seats[i] for i in range(len(entitled_seats))]

# Calculate the total seats allocated initially
total_initial_seats = sum(initial_seats)

# Distribute remaining seats based on the largest remainders
remaining_seats = SEATS - total_initial_seats
for _ in range(remaining_seats):
    max_remainder_index = remainders.index(max(remainders))
    initial_seats[max_remainder_index] += 1
    remainders[max_remainder_index] = 0  # Set to 0 to avoid re-allocating the same seat

# Adjust for overhang seats
final_seats = initial_seats.copy()
for i in range(len(final_seats)):
    if FPP_results[i] > final_seats[i]:
        final_seats[i] = FPP_results[i]

# Scale up the seats proportionally if needed
total_seats = sum(final_seats)
while total_seats < SEATS:
    min_difference = min([entitled_seats[i] - final_seats[i] for i in range(len(final_seats)) if entitled_seats[i] > final_seats[i]])
    min_index = [entitled_seats[i] - final_seats[i] for i in range(len(final_seats))].index(min_difference)
    
    final_seats[min_index] += 1
    total_seats += 1

print("Final Seats:", final_seats)
