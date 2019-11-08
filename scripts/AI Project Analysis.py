# AI Project - Analysis
# Christopher Cruzen
# 11.07.2019
#
# This program contains the machine learning analysis conducted on star system FLUX
# hada for my final AI project. Data is pre-split and hosted below.
#
# Raw Source: https://www.kaggle.com/cacruzen/star-system-flux-metadata
#
# Note: This code was run through kaggle's provided notebook editor,
#       which simplified the dataset import process. Link:
#       https://www.kaggle.com/cacruzen/ai-project-analysis

# Imports
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Retrieve Datasets
train = np.genfromtxt(
    '/kaggle/input/star-system-flux-metadata/train.csv',
    delimiter=','
)
validation = np.genfromtxt(
    '/kaggle/input/star-system-flux-metadata/validation.csv',
    delimiter=','
)
test = np.genfromtxt(
    '/kaggle/input/star-system-flux-metadata/test.csv',
    delimiter=','
)

# Shuffle Datasets
np.random.shuffle(train)
np.random.shuffle(validation)
np.random.shuffle(test)

# Plot Inputs Against Outputs (#1 - DONE)
# plt.figure(figsize=(5,5))
# plt.scatter(train[:, 1], train[:, 0], color='b', alpha=0.2)
# plt.xlabel('Average FLUX')
# plt.ylabel('Planetary (0=N, 1=Y)')
# plt.show()
# plt.figure(figsize=(5,5))
# plt.scatter(train[:, 1], train[:, 1], color='b', alpha=0.2)
# plt.xlabel('Median FLUX')
# plt.ylabel('Planetary (0=N, 1=Y)')
# plt.show()
# plt.figure(figsize=(5,5))
# plt.scatter(train[:, 1], train[:, 2], color='b', alpha=0.2)
# plt.xlabel('Maximum FLUX')
# plt.ylabel('Planetary (0=N, 1=Y)')
# plt.show()
# plt.figure(figsize=(5,5))
# plt.scatter(train[:, 1], train[:, 3], color='b', alpha=0.2)
# plt.xlabel('Mainimum FLUX')
# plt.ylabel('Planetary (0=N, 1=Y)')
# plt.show()
# plt.figure(figsize=(5,5))
# plt.scatter(train[:, 1], train[:, 4], color='b', alpha=0.2)
# plt.xlabel('Std. Dev. FLUX')
# plt.ylabel('Planetary (0=N, 1=Y)')
# plt.show()

# Create Regression Model
model = Sequential()

# Split Datasets
train_input = train[:, 1:5]
train_output = train[:, 0:1]
validation_input = validation[:, 1:5]
validation_output = validation[:, 0:1]
test_input = test[:, 1:5]
test_output = test[:, 0:1]

# Add Neural Network Layers
model.add(Dense(5, input_dim=len(train_input[0]), activation='linear'))
model.add(Dense(4, input_dim=len(train_input[0]), activation='linear'))
model.add(Dense(2, input_dim=len(train_input[0]), activation='linear'))
model.add(Dense(1, input_dim=len(train_input[0]), activation='linear'))

# Print Summary
print(model.summary())

# Compile Model w/ Loss="Mean Absolute Error" and Optmizer="Stochastic Gradient Descent"
loss_method = 'mae' # "Mean Absolute Error"
model.compile(loss=loss_method, optimizer='sgd', metrics=[loss_method, 'accuracy'])

# Perform Fit
history = model.fit(
    train_input,
    train_output,
    epochs=25,
    verbose=0,
    batch_size=10,
    validation_data = (
        validation_input,
        validation_output
    )
)

# Determine Maximum Error
max_error = 0
for error in history.history[loss_method]:
    if max_error < error:
        max_error = error
        
# Plot Results
plt.figure(figsize=(5,5))
plt.plot(history.history[loss_method])
plt.plot(history.history['val_' + loss_method])
plt.ylabel('Mean Absolute Error')
plt.ylim(0,max_error)
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Read Weights
# w0 = model.layers[0].get_weights()[0][0]
# w1 = model.layers[0].get_weights()[0][1]
# w2 = model.layers[0].get_weights()[0][2]
# w3 = model.layers[0].get_weights()[0][3]
# w4 = model.layers[0].get_weights()[0][4]
# w5 = model.layers[0].get_weights()[0][5]
# b1 = model.layers[0].get_weights()[1]

# Evaluate Model w/ Test Dataset
test_predictions = model.predict(test_input)
actual_preview = test_output[0:5]
predict_preview = test_predictions[0:5]
print("Model Output Preview (Predicted Actual):")
for index in range(len(actual_preview)):
    print("\t%.4f" % predict_preview[index][0]
        + "\t" + str(actual_preview[index][0])
    )

# Evaluate Model
model.evaluate(x=test_input, y=test_output)