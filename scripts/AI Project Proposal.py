# AI Project Proposal - Dataset Analysis
# Christopher Cruzen
# 09.29.2019
#
# Note: This code was run through kaggle's provided notebook editor,
#       which somewhat simplified the data import process. Link:
#       https://www.kaggle.com/cacruzen/ai-project-proposal/edit

import numpy as np # linear algebra
import matplotlib.pyplot as plt

# Declare Constants
SAMPLE_ROW = 25 # Row 0 Contains Headers
SAMPLE_COLUMNS = 40

# Retrieve Dataset
dataset = np.genfromtxt(
    '/kaggle/input/kepler-labelled-time-series-data/exoTrain.csv',
    delimiter=',',
    usecols=(range(SAMPLE_COLUMNS + 1)) # Columns 1-11, Total Number of Columns: 3198
)

# Print Dataset Dimensions
print(dataset.shape)

# Extract Column Data
star_system_flux_values = dataset[SAMPLE_ROW,1:] # Get Flux Values for System
star_system_flux_items = np.arange(len(star_system_flux_values))

# Plot Column
plt.bar(star_system_flux_items, star_system_flux_values, align='center', alpha=.5)
plt.xticks(np.arange(0, SAMPLE_COLUMNS, 2))
plt.title('System ' + str(SAMPLE_ROW) + ' Preview')
plt.ylabel('FLUX Value')
plt.xlabel('Record')
plt.show()