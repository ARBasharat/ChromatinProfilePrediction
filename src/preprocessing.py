import h5py
import numpy as np
import scipy.io
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def hot2seq(hot_array):
  data = ['A', 'G', 'C', 'T']
  values = np.array(data)
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(values)
  inte_array = np.einsum('ijk,k->ij', hot_array, np.array([0,1,2,3], dtype=np.uint8))
  converted_array = []
  for idx in range(0, inte_array.shape[0]):
    tmp = label_encoder.inverse_transform(inte_array[idx])
    converted_array.append(''.join(tmp.astype(str)))
  return np.array(converted_array)

## Process Training data -- write hdf5 file
trainmat = h5py.File('../data_seq/train.mat')
N = trainmat['trainxdata'].shape[-1] 
c = trainmat['traindata'].shape[0] 
dt = h5py.special_dtype(vlen=str)
train_h5 = h5py.File('../data_seq/train.hdf5', 'w')
X_h5 = train_h5.create_dataset("X_train", (N,), dt)
y_h5 = train_h5.create_dataset("y_train", (N,c), 'u1')
chunk_sz = 200_000
n_chunks = N//chunk_sz if N%chunk_sz==0 else N//chunk_sz+1
for i in range(n_chunks):
  fi = int( i   *chunk_sz)
  ti = int((i+1)*chunk_sz) if i!=(n_chunks-1) else N
  X_train = np.transpose(trainmat['trainxdata'][:,:,fi:ti],axes=(2,0,1))
  y_train = (trainmat['traindata'][:,fi:ti]).T
  X_train = hot2seq(X_train)
  X_h5[fi:ti] = X_train
  y_h5[fi:ti] = y_train
train_h5.close()

## Process Validation Data -- write npy file
validmat = scipy.io.loadmat('../data_seq/valid.mat')
X_val = np.transpose(validmat['validxdata'],axes=(0,2,1))
y_val = validmat['validdata']
X_val = hot2seq(X_val)
np.savez_compressed('../data_seq/valid', X_val, y_val)

## Process Test Data -- write npy file 
testmat = scipy.io.loadmat('../data_seq/test.mat')
X_test = np.transpose(testmat['testxdata'],axes=(0,2,1))
y_test = testmat['testdata']
X_test  = hot2seq(X_test)
np.savez_compressed('../data_seq/test', X_test, y_test)
