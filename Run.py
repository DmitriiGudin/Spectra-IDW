from __future__ import division
import numpy as np
import h5py
import time
import get_interpolated_spectrum as g_i_s
import params


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


if __name__ == '__main__':
    start_time = time.time()
    def Print(text):
        print time.time()-start_time, "s: ", text
    Teff = get_column(0, float, params.param_vals_file)
    logg = get_column(1, float, params.param_vals_file)
    FeH = get_column(2, float, params.param_vals_file)
    CFe = get_column(3, float, params.param_vals_file)
    Print ("Program started. "+str(len(Teff))+" spectra to generate.")
    Print ("Generating...")

    spectra_list = g_i_s.get_interpolated_spectra(Teff, logg, FeH, CFe)
    for i, s in enumerate(spectra_list):
        filename = params.output_directory + "/interpolated_spectrum_" + str(Teff[i])[:min(len(str(Teff[i]))+1,10)] + "_" + str(logg[i])[:min(len(str(logg[i]))+1,10)] + "_" + str(FeH[i])[:min(len(str(FeH[i]))+1,10)] + "_" + str(CFe[i])[:min(len(str(CFe[i]))+1,10)] + ".hdf5"
        f = h5py.File(filename, 'w')
        f.create_dataset("/T_EFF", (1,1), dtype='f')
        f.create_dataset("/LOG_G", (1,1), dtype='f')
        f.create_dataset("/FE_H", (1,1), dtype='f')
        f.create_dataset("/C_FE", (1,1), dtype='f')
        f.create_dataset("/spectrum", (1,params.wavelength_range[1]-params.wavelength_range[0]+1), dtype='f')

        f["/T_EFF"][0] = Teff[i]
        f["/LOG_G"][0] = logg[i]
        f["/FE_H"][0] = FeH[i]
        f["/C_FE"][0] = CFe[i]
        f["/spectrum"][0] = s

    Print ("Done.")
    f.close()
