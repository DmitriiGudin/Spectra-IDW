# Written by Dmitrii Gudin, U Notre Dame.


from __future__ import division 
import numpy as np
import h5py
import IDW
import Linear
import params


# Returns True is the spectrum doesn't have false "spikes" and negative values, and False if it does and cannot be used for interpolation.
def is_good (s):
    # Calculate the maximum flux value of the entire spectrum.
    max_val = max(s)
    # Calculate the maximum flux value of the main body of the spectrum.
    max_val_lower = max(np.sort(s)[:int((0.9)*len(s))])
    # Return the result of the comparison.
    return (max_val/max_val_lower<5) and (min(s)>0)


# Obtains spectrum from the spectral grid interpolation. The IDW method is used.
#
# Teff - Effective temperature.
# logg - Logarithmic surface gravity.
# FeH - [Fe/H], metallicity.
# CFe - [C/Fe], carbonicity.
#
# Returns the flux array of the interpolated spectrum.


def get_interpolated_spectrum (Teff, logg, FeH, CFe):
    f = h5py.File(params.spectral_grid_file, 'r')
    T_EFF = f["/T_EFF"][:].flatten()
    LOG_G = f["/LOG_G"][:].flatten()
    FE_H = f["/FE_H"][:].flatten()
    C_FE = f["/A_C"][:].flatten()
    spectrum = f["/spectrum"][:]
    f.close()
    
    if params.remove_bad_spectra:
        good_indeces = []
        for i in range(len(spectrum)):
            if is_good(spectrum[i]):
                good_indeces.append(i)
        T_EFF = T_EFF[good_indeces]
        LOG_G = LOG_G[good_indeces]
        FE_H = FE_H[good_indeces]
        C_FE = C_FE[good_indeces]
        spectrum = spectrum[good_indeces]

    if params.method == 'IDW':
        return IDW.IDW (np.array([[Teff,logg,FeH,CFe]]), np.transpose(np.array([T_EFF,LOG_G,FE_H,C_FE])), spectrum, Shepard_parameter=params.Shepard_parameter, N_points=params.N_nearest_points)
    elif params.method == 'Linear':
        return Linear.Linear (np.array([[Teff,logg,FeH,CFe]]), np.transpose(np.array([T_EFF,LOG_G,FE_H,C_FE])), spectrum, N_points=params.N_nearest_points, method=params.GD_method)


# Obtains multiple spectra from the spectral grid interpolation. The IDW method is used.
#
# Teff_arr - Effective temperature - array.
# logg_arr - Logarithmic surface gravity - array.
# FeH_arr - [Fe/H], metallicity - array.
# CFe_arr - [C/Fe], carbonicity - array.
#
# Returns the list of flux arrays of the interpolated spectra.


def get_interpolated_spectra (Teff_arr, logg_arr, FeH_arr, CFe_arr):
    f = h5py.File(params.spectral_grid_file, 'r')
    T_EFF = f["/T_EFF"][:].flatten()
    LOG_G = f["/LOG_G"][:].flatten()
    FE_H = f["/FE_H"][:].flatten()
    C_FE = f["/A_C"][:].flatten()
    spectrum = f["/spectrum"][:]
    f.close()
    
    if params.remove_bad_spectra:
        good_indeces = []
        for i in range(len(spectrum)):
            if is_good(spectrum[i]):
                good_indeces.append(i)
        T_EFF = T_EFF[good_indeces]
        LOG_G = LOG_G[good_indeces]
        FE_H = FE_H[good_indeces]
        C_FE = C_FE[good_indeces]
        spectrum = spectrum[good_indeces]

    vars = np.transpose(np.array([T_EFF, LOG_G, FE_H, C_FE]))
    new_vars = np.transpose(np.array([Teff_arr, logg_arr, FeH_arr, CFe_arr]))
    if params.method == 'IDW':
        return IDW.IDW (new_vars, vars, spectrum, Shepard_parameter=params.Shepard_parameter, N_points=params.N_nearest_points)
    elif params.method == 'Linear':
        return Linear.Linear (new_vars, vars, spectrum, N_points=params.N_nearest_points)

