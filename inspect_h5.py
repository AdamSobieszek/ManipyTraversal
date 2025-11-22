import h5py

file_path = 'data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
try:
    with h5py.File(file_path, 'r') as f:
        print("Keys in HDF5 file:")
        print(list(f.keys()))
except Exception as e:
    print(f"Error reading HDF5 file: {e}")
