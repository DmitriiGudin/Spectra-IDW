# Written by Dmitrii Gudin, U Notre Dame.


from __future__ import division
import numpy as np


# Inverse distance weighting interpolation implementation in Python. All numbers and function values are treated as multidimensional floats (tuples).
#
# new_var_values: an array of coordinates to calculate function values on.
# var_values: an array of coordinates with known function values.
# func_values: an array of known function values.
# Shepard_parameter: the exponent value d of the weighting function: w = 1 / Distance^d.
#
# Returns a multidimensional function value.

def IDW (new_var_values, var_values, func_values, Shepard_parameter=1):
    new_var_values, var_values, func_values = np.array(new_var_values), np.array(var_values), np.array(func_values)
    # Normalize the coordinates.
    var_mean, var_std = np.mean(var_values), np.std(var_values)
    new_var_values, var_values = (new_var_values-var_mean)/var_std, (var_values-var_mean)/var_std
    # Calculate the function values.
    new_func_values = np.zeros((len(new_var_values),len(func_values[0])))
    for i in range(len(new_func_values)): 
        if new_var_values[i] in var_values:
            new_func_values[i] = func_values[np.where(np.all(var_values==new_var_values[i],axis=1))[0][0]]
        else:
            distances = np.zeros((len(var_values)))
            for j in range(len(distances)):
                distances[j] = np.sqrt(np.dot((new_var_values[i]-var_values[j]),(new_var_values[i]-var_values[j])))
            weights = 1 / (distances**Shepard_parameter)
            for j in range(len(func_values[0])):
                new_func_values[i,j] = np.dot((np.transpose(func_values))[j],weights)/sum(weights)
    # Return the function values.
    return new_func_values

