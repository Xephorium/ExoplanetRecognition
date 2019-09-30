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
#   4. Mode FLUX
#   5. Max FLUX
#   6. Min FLUX
#   7. Std. Dev. FLUX
#
# Raw Source: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data
#
# Note: This code was run through kaggle's provided notebook editor,
#       which simplified the dataset import process. Link:
#       https://www.kaggle.com/cacruzen/ai-project-proposal

import numpy as np
import matplotlib.pyplot as plt

# Declare Constants
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

    # Compute System Metadata (Planetary, Mean, Median, Mode, Max, Min, Standard Deviation)
    star_system_metadata = [
        dataset[current_system,0] - 1,                          # Planetary (0,1)
        round(np.average(star_system_flux_values), 2),          # Average FLUX
        round(np.median(star_system_flux_values), 2),           # Median FLUX
        round(np.amax(star_system_flux_values), 2),             # Max FLUX
        round(np.amin(star_system_flux_values), 2),             # Min FLUX    DUP??
        round(np.std(star_system_flux_values), 2)               # Std. Dev. FLUX
    ]
    
    # Normalize Metadata
    normalized_data = [star_system_metadata[0]]
    min = star_system_metadata[1]
    max = star_system_metadata[1]
    for record in range(1, len(star_system_metadata)):
        if star_system_metadata[record] < min:
            min = star_system_metadata[record]
        if star_system_metadata[record] > max:
            max = star_system_metadata[record]
    for record in range(1, len(star_system_metadata)):
        if max - min != 0:
            normalized_data.append(round((star_system_metadata[record] - min)/(max - min), 4))
        else:
            normalized_data.append(0.0)
    
    # Save Metadata
    system_metadata.append(normalized_data)

# Print Records
for system in range(0, len(system_metadata)):
    for value in range(0, 6):
        print(str(system_metadata[system][value]), end='')
        if value != 5:
            print(',', end='')
    print('')