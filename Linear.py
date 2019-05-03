# Written by Dmitrii Gudin, U Notre Dame.


from __future__ import division
import numpy as np
from scipy.interpolate import LinearNDInterpolator


# Linear interpolation implementation in Python. All numbers and function values are treated as multidimensional floats (tuples).
#
# new_var_values: an array of coordinates to calculate function values on.
# var_values: an array of coordinates with known function values.
# func_values: an array of known function values.
# N_points: number of the closest points to interpolate over. 0 means that all points are involved.
#
# Returns a multidimensional function value.


def get_closest_points (point, points, N):
    points = points - point
    points = sorted(points, key = lambda N: np.dot(N,N))
    return points[:N]+point


def Linear (new_var_values, var_values, func_values, N_points = 0):
    new_var_values, var_values, func_values = np.array(new_var_values), np.array(var_values), np.array(func_values)
    # Normalize the coordinates.
    var_mean, var_std = np.mean(var_values), np.std(var_values)
    new_var_values, var_values = (new_var_values-var_mean)/var_std, (var_values-var_mean)/var_std
    # Calculate the function values.
    new_func_values = np.zeros((len(new_var_values),len(func_values[0])))
    for i in range(len(new_func_values)): 
        if list(new_var_values[i]) in var_values.tolist():
            new_func_values[i] = func_values[np.where(np.all(var_values==new_var_values[i],axis=1))[0][0]]
        else:
            if N_points == 0:
                temp_var_values = var_values 
                temp_func_values = func_values
            elif N_points > 0: 
                temp_var_values = get_closest_points (new_var_values[i], var_values, min(N_points,len(var_values)))
                temp_func_values = np.array([func_values[np.where(np.all(var_values==t,axis=1))[0][0]] for t in temp_var_values])
            linInter = LinearNDInterpolator(temp_var_values, temp_func_values)
            new_func_values[i] = linInter(new_var_values[i])
    # Return the function values.
    return new_func_values

