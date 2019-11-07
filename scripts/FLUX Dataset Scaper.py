# AI Project Proposal - FLUX Dataset Scraper
# Christopher Cruzen
# 09.29.2019
#
# This program scrapes the star system luminocity dataset provided at
# the link below and condenses 3198 FLUX input variables per star
# system into a much more managable 7 metadata variables for machine
# learning inputs/outputs. The resulting FLUX summary variables are:
#
#   1. Planetary (0,1)
#   2. Average FLUX
#   3. Median FLUX
#   4. Max FLUX
#   5. Min FLUX
#   6. Std. Dev. FLUX
#
# Raw Source: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data
#
# Note: This code was run through kaggle's provided notebook editor,
#       which simplified the dataset import process. Link:
#       https://www.kaggle.com/cacruzen/ai-project-proposal

import numpy as np
import matplotlib.pyplot as plt

# Declare Constants
DECIMAL_ACCURACY = 7
SAMPLE_ROWS = 5087 # Max = Total (Total Number of Rows: 5087)
SAMPLE_COLUMNS = 3197 # Max = Total - 1 (Total Number of Columns: 3198)

# Declare Variables
system_metadata = []

# Retrieve Dataset
dataset = np.genfromtxt(
    '/kaggle/input/kepler-labelled-time-series-data/exoTrain.csv',
    delimiter=',',
    usecols=(range(SAMPLE_COLUMNS + 1))
)

# Iterate Through Systems
for current_system in range(1, SAMPLE_ROWS + 1):

    # Extract System Data
    star_system_flux_values = dataset[current_system,1:] # Get Flux Values for System

    # Compute System Metadata (Planetary, Mean, Median, Max, Min, Standard Deviation)
    star_system_metadata = [
        dataset[current_system,0] - 1,                                         # Planetary (0,1)
        round(np.average(star_system_flux_values), DECIMAL_ACCURACY),          # Average FLUX
        round(np.median(star_system_flux_values), DECIMAL_ACCURACY),           # Median FLUX
        round(np.amax(star_system_flux_values), DECIMAL_ACCURACY),             # Max FLUX
        round(np.amin(star_system_flux_values), DECIMAL_ACCURACY),             # Min FLUX    DUP??
        round(np.std(star_system_flux_values), DECIMAL_ACCURACY)               # Std. Dev. FLUX
    ]
    
    # Normalize Mean, Median, Max, & Min Relative to System Values
    normalized_data = [star_system_metadata[0]]
    min = star_system_metadata[1]
    max = star_system_metadata[1]
    for record in range(1, len(star_system_metadata) - 1):
        if star_system_metadata[record] < min:
            min = star_system_metadata[record]
        if star_system_metadata[record] > max:
            max = star_system_metadata[record]
    for record in range(1, len(star_system_metadata) - 1):
        if max - min != 0:
            if star_system_metadata[record] != max and star_system_metadata[record] != min: 
                normalized_data.append(round((star_system_metadata[record] - min)/(max - min), DECIMAL_ACCURACY))
            else:
                normalized_data.append(star_system_metadata[record])
        else:
            normalized_data.append(0.0)
    
    # Save Metadata
    system_metadata.append(normalized_data)
    
# Normalize Max Relative to Dataset Values
min = dataset[1][3]
max = dataset[1][3]
for current_system in range(1, SAMPLE_ROWS + 1):
    if dataset[current_system][3] < min:
        min = dataset[current_system][3]
    if dataset[current_system][3] > max:
        max = dataset[current_system][3]
for current_system in range(1, SAMPLE_ROWS + 1):
    if max - min != 0:
        system_metadata[current_system - 1][3] = round((dataset[current_system][3] - min)/(max - min), DECIMAL_ACCURACY)
    else:
        system_metadata[current_system - 1][3] = 0.0
        
# Normalize Max Relative to Dataset Values
min = dataset[1][4]
max = dataset[1][4]
for current_system in range(1, SAMPLE_ROWS + 1):
    if dataset[current_system][4] < min:
        min = dataset[current_system][4]
    if dataset[current_system][4] > max:
        max = dataset[current_system][4]
for current_system in range(1, SAMPLE_ROWS + 1):
    if max - min != 0:
        system_metadata[current_system - 1][4] = round((dataset[current_system][4] - min)/(max - min), DECIMAL_ACCURACY)
    else:
        system_metadata[current_system - 1][4] = 0.0

    
# Normalize Standard Deviation Relative to Dataset Values
min = dataset[1][5]
max = dataset[1][5]
for current_system in range(1, SAMPLE_ROWS + 1):
    if dataset[current_system][5] < min:
        min = dataset[current_system][5]
    if dataset[current_system][5] > max:
        max = dataset[current_system][5]
for current_system in range(1, SAMPLE_ROWS + 1):
    if max - min != 0:
        system_metadata[current_system - 1].append(round((dataset[current_system][5] - min)/(max - min), DECIMAL_ACCURACY))
    else:
        system_metadata[current_system - 1].append(0.0)

# Print Records
for system in range(0, len(system_metadata)):
    for value in range(0, 6):
        print(str(system_metadata[system][value]), end='')
        if value != 5:
            print(',', end='')
    print('')