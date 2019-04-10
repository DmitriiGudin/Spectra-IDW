from __future__ import division
import numpy as np
import h5py
import get_interpolated_spectrum as g_i_s
import params


prompts = ["Teff", "log(g)", "[Fe/H]", "[C/Fe]"]


if __name__ == '__main__':
    Vars = [0,0,0,0]
    for i, p in enumerate(prompts):
        Vars[i] = float(raw_input (p+" = \n"))
    Teff, logg, FeH, CFe = Vars
    filename = params.output_directory + "/interpolated_spectrum_" + str(Teff)[:min(len(str(Teff))+1,10)] + "_" + str(logg)[:min(len(str(logg))+1,10)] + "_" + str(FeH)[:min(len(str(FeH))+1,10)] + "_" + str(CFe)[:min(len(str(CFe))+1,10)] + ".hdf5"
    print filename
    f = h5py.File(filename, 'w')
    f.create_dataset("/T_EFF", (1,1), dtype='f')
    f.create_dataset("/LOG_G", (1,1), dtype='f')
    f.create_dataset("/FE_H", (1,1), dtype='f')
    f.create_dataset("/C_FE", (1,1), dtype='f')
    f.create_dataset("/spectrum", (1,7001), dtype='f')

    f["/T_EFF"][0] = Teff
    f["/LOG_G"][0] = logg
    f["/FE_H"][0] = FeH
    f["/C_FE"][0] = CFe
    f["/spectrum"][0] = g_i_s.get_interpolated_spectrum(Teff, logg, FeH, CFe)

    f.close()
