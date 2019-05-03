from __future__ import division
import numpy as np
import h5py
import params

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Train and cross-validation batch generator.
def batch_gen (data_x, data_y):
    while True:
        # Generate a random set of indeces.
        indeces = random.sample(range(0,len(data_x)),1)
        # Yield the data for those indeces, reshaping it properly.
        yield (np.array([data_x[i] for i in indeces]).reshape(1, len(data_x[0]), 1), np.array([data_y[i] for i in indeces]))


# Creates and sets up a CNN model with parameters specified in params.py.
def create_model():
    return Sequential([InputLayer(batch_input_shape=(1, 4, 1)),
    Conv1D(kernel_initializer=params.initializer, activation=params.activation, padding="same", filters=params.num_filters[0], kernel_size=params.filter_length[0]),
    Conv1D(kernel_initializer=params.initializer, activation=params.activation, padding="same", filters=params.num_filters[1], kernel_size=params.filter_length[1]),
    MaxPooling1D(pool_size=params.maxpooling_length),
    Flatten(),
    Dense(units=params.num_hidden_layers[0], kernel_initializer=params.initializer, activation=params.activation),
    Dense(units=params.num_hidden_layers[1], kernel_initializer=params.initializer, activation=params.activation),
    Dense(units=4, activation="linear", input_dim=params.num_hidden_layers[1]),
])


# Creates and sets up an optimizer for the CNN model with parameters specified in params.py.
def create_optimizer():
    return Adam(lr=params.lr, beta_1=params.beta_1, beta_2=params.beta_2, epsilon=params.optimizer_epsilon, decay=0.0)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=params.early_stopping_min_delta, 
    patience=params.early_stopping_patience, verbose=2, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=params.reduce_lr_epsilon, 
    patience=params.reduce_lr_patience, min_lr=params.reduce_lr_min, mode='min', verbose=2)

if __name__ == '__main__':
    # Load the data file.
    File = h5py.File(params.spectral_grid_file, 'r')
    # Normalize the variables and prepare them for training.
    data_x = np.column_stack((((File['T_EFF'][:]-T_EFF_mean)/T_EFF_std)[:],((File['LOG_G'][:]-LOG_G_mean)/LOG_G_std)[:],((File['FE_H'][:]-FE_H_mean)/FE_H_std)[:],((File['C_FE'][:]-C_FE_mean)/C_FE_std)[:]))
    data_y = np.divide(File['spectrum'][:]-spectrum_mean,spectrum_std)[:]
    # How many training/cross-validation spectra do we have?
    N_train = int(len(data_y)*(1-params.cv_data_frac))
    # Split the data into training/cross-validation data and clean up the obsolete data.
    data_x_train, data_y_train = data_x[0:N_train], data_y[0:N_train]
    data_x_cv, data_y_cv = data_x[N_train:N_spectra], data_y[N_train:N_spectra]
    # Generate CNN model.
    batch_model = create_model(params.batch_size)
    # Generate optimizer for the model and some Keras functions.
    optimizer = create_optimizer()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=params.early_stopping_min_delta, patience=params.early_stopping_patience, verbose=2, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=params.reduce_lr_epsilon, patience=params.reduce_lr_patience, min_lr=params.reduce_lr_min, mode='min', verbose=2)
    # Compile and save the model.
    model = create_model(1)
    model.set_weights(batch_model.get_weights())
    model.compile(optimizer=optimizer, loss=params.loss_function, metrics=params.metrics)
    model.save(params.CNN_FILE)
    print "Training done."
