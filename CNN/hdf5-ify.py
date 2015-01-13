import os
import csv
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

multiclass = True

# Set paths to input data .csv files
inDirname = os.path.abspath('./csv_data')
X_csv = os.path.join(inDirname, 'train/sensorActivation.csv')
y_csv = os.path.join(inDirname, 'train/concentration.csv')
Xt_csv = os.path.join(inDirname, 'test/sensorActivation.csv')
yt_csv = os.path.join(inDirname, 'test/concentration.csv')

# read the data into numpy arrays
reader = csv.reader(open(X_csv,"rb"), delimiter=",")
csvIn = list(reader)
X = np.array(csvIn).astype('float')

reader = csv.reader(open(y_csv,"rb"), delimiter=",")
csvIn = list(reader)
y = np.array(csvIn).astype('float')

reader = csv.reader(open(Xt_csv,"rb"), delimiter=",")
csvIn = list(reader)
Xt = np.array(csvIn).astype('float')

reader = csv.reader(open(yt_csv,"rb"), delimiter=",")
csvIn = list(reader)
yt = np.array(csvIn).astype('float')

#standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)
Xt = scaler.transform(Xt)

# convert to multiclass dataset
if (multiclass):
	y = np.argmax(y[:,1:5], axis=1)
	yt = np.argmax(yt[:,1:5], axis=1)

# Write out the data to HDF5 files in a temp directory.
# This file is assumed to be caffe_root/examples/hdf5_classification.ipynb
dirname = os.path.abspath('./hdf5_data')
if not os.path.exists(dirname):
    os.makedirs(dirname)

train_filename = os.path.join(dirname, 'train.h5')
test_filename = os.path.join(dirname, 'test.h5')

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
with h5py.File(train_filename, 'w') as f:
    f['data'] = X
    f['label'] = y.astype(np.float32)
with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')
    
# HDF5 is pretty efficient, but can be further compressed.
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')
