from __future__ import division
import numpy as np


# GENERAL PARAMETERS
remove_bad_spectra = False
Shepard_parameter = 2
N_nearest_points = 100
wavelength_range = [3000, 5000]
method = 'IDW' # 'IDW' or 'Linear'


# FILE PARAMETERS
spectral_grid_file = "R2000_clean_normed_cut.hdf5"
param_vals_file = "param_vals.csv"
CNN_file = "CNN.hdf5"
output_directory = "output"


# NEURAL NETWORK PARAMETERS
S_N = 20 # Signal-to-noise ratio for noise injection.
cv_data_frac = 0.05 # What fraction of the training data to use for cross-validation. The rest is used for training.
N_spectra = 0 # Number of spectra to train/test the model on. Non-positive means working with the whole dataset. Mostly for short testing purposes.
activation = 'relu' # Activation function type for hidden layers.
initializer = 'he_normal' # How to calculate the initial model weights.
num_hidden_layers = [256, 128] # Number of neurons in hidden layers in the CNN.
num_filters = [4, 16] # Number of filters in colvolutional layers in the CNN.
filter_length = [8, 8] # Length of filters.
maxpooling_length = 4 # Length of the maxplooling window.
max_epochs = 100 # Number of training epochs (loops over the entire data).
lr, beta_1, beta_2, optimizer_epsilon = 0.0007, 0.9, 0.999, 1e-08 # Adam optimizer parameters.
early_stopping_min_delta, early_stopping_patience, reduce_lr_factor, reduce_lr_epsilon, reduce_lr_patience, reduce_lr_min, loss_function, metrics = 0.0001, 4, 0.5, 0.0009, 2, 0.00008, 'mean_squared_error', ['accuracy', 'mae'] # Loss function parameters.
