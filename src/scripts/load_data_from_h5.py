import h5py
import os


def load_cifs_from_hdf5(hdf5_filepath):
    cif_data = {}
    with h5py.File(hdf5_filepath, 'r') as f:
        for key in f.keys():
            if key.lower().endswith('.cif'):
                cif_content = f[key][()]
                if isinstance(cif_content, bytes):
                    cif_content = cif_content.decode('utf-8')
                cif_data[key] = cif_content
    return cif_data
