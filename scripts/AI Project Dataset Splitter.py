# AI Project - Dataset Splitter
# Christopher Cruzen
# 11.07.2019
#
# This program takes the clean, normalized FLUX metadata set generated
# for my project proposal and splits it into three categories: train,
# validation, and test.
#
# Raw Source: https://www.kaggle.com/cacruzen/star-system-flux-metadata
#
# Note: This code was run through kaggle's provided notebook editor,
#       which simplified the dataset import process. Link:
#       https://www.kaggle.com/cacruzen/ai-project-dataset-splitter

import numpy as np
import matplotlib.pyplot as plt

# Declare Constants
DECIMAL_ACCURACY = 7
TRAIN_SYSTEMS = 4000
TRAIN_EXOPLANETS = 25
VALIDATION_SYSTEMS = 1000
VALIDATION_EXOPLANETS = 9
TEST_SYSTEMS = 657
TEST_EXOPLANETS = 8

# Delcare Initial Split Variables
systems_barren = []
systems_exoplanet = []

# Declare Final Split Variables
train = []
validation = []
test = []

# Retrieve Dataset
dataset = np.genfromtxt(
    '/kaggle/input/star-system-flux-metadata/Combined.csv',
    delimiter=','
)

# Split Systems Into Baren and Exoplanet
for system in dataset:
    if system[0] == 1:
        systems_exoplanet.append(system)
    else:
        systems_barren.append(system)

# Shuffle Initial Split
np.random.shuffle(systems_exoplanet)
np.random.shuffle(systems_barren)

# Split Train Set
for index in range(TRAIN_EXOPLANETS):
    train.append(systems_exoplanet[0])
    del systems_exoplanet[0]
for index in range(TRAIN_SYSTEMS - TRAIN_EXOPLANETS):
    train.append(systems_barren[0])
    del systems_barren[0]
    
# Split Validation Set
for index in range(VALIDATION_EXOPLANETS):
    validation.append(systems_exoplanet[0])
    del systems_exoplanet[0]
for index in range(VALIDATION_SYSTEMS - VALIDATION_EXOPLANETS):
    validation.append(systems_barren[0])
    del systems_barren[0]

# Split Test Set
for index in range(TEST_EXOPLANETS):
    test.append(systems_exoplanet[0])
    del systems_exoplanet[0]
for index in range(TEST_SYSTEMS - TEST_EXOPLANETS):
    test.append(systems_barren[0])
    del systems_barren[0]

    
# Print Datasets

print("--- Train ---")
for system in train:
    print("%.1f" % system[0] + ", %.7f" % system[1] + ", %.7f" % system[2] + ", %.7f" % system[3] + ", %.7f" % system[4] + ", %.7f" % system[5])
    
print("--- Validation ---")
for system in validation:
    print("%.1f" % system[0] + ", %.7f" % system[1] + ", %.7f" % system[2] + ", %.7f" % system[3] + ", %.7f" % system[4] + ", %.7f" % system[5])

print("--- Test ---")
for system in test:
    print("%.1f" % system[0] + ", %.7f" % system[1] + ", %.7f" % system[2] + ", %.7f" % system[3] + ", %.7f" % system[4] + ", %.7f" % system[5])