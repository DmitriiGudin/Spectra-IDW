# Created by Dmitrii Gudin, U Notre Dame.

from __future__ import division
import numpy as np
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt


wavelength_array = np.arange (3000, 5000+1)


def plot_stuff(f):
    Spectrum, T_EFF, LOG_G, FE_H, C_FE = f['/spectrum'][0], f['/T_EFF'][0][0], f['/LOG_G'][0][0], f['/FE_H'][0][0], f['/C_FE'][0][0]
    plt.clf()
    plt.title("Interpolated spectrum : Teff = "+str(T_EFF)+", log(g) = "+str(LOG_G)+", [Fe/H] = "+str(FE_H)+", [C/Fe] = "+str(C_FE), size=24)
    plt.xlabel("Wavelength (A)", size=24)
    plt.ylabel("Flux", size=24)
    plt.tick_params(labelsize=18)
    plt.xlim(min(wavelength_array), max(wavelength_array))
    plt.ylim(min(Spectrum), max(Spectrum))
    plt.plot(wavelength_array, Spectrum, linewidth=2, color='black')
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.show()


if __name__ == '__main__':
    filename = str(sys.argv[1])
    f = h5py.File (filename, 'r')
    plot_stuff(f)
