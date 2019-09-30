# AI Project Proposal - Normalized Dataset Analysis
# Christopher Cruzen
# 09.29.2019
#
# This program converts normalized star system FLUX data
# into a number of useful visualizations for my machine
# learning project proposal. 

import numpy as np
import matplotlib.pyplot as plt

# Retrieve Dataset
dataset = np.genfromtxt(
    'https://raw.githubusercontent.com/Xephorium/ExoplanetRecognition/master/datasets/Train.csv',
    delimiter=","
)

# Plot Aggregate Column Data
for column in range(1, 6):
  
  # Get Column
  values = dataset[0:,column]
  
  # Plot Histogram
  plt.hist(values, 30, alpha=.5)
  plt.ylabel('Occurrences')
  
  # Set Title
  title = ''
  if (column == 1):
    title = 'Normalized Mean FLUX'
  elif (column == 2):
    title = 'Normalized Median FLUX'
  elif (column == 3):
    title = 'Normalized Max FLUX'
  elif (column == 4):
    title = 'Normalized Min FLUX'
  else:
    title = 'Normalized Std. Dev. of FLUX'
  plt.xlabel(title)
  
  # Display Graph
  plt.show()