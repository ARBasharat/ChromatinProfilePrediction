import h5py
import numpy as np
import scipy.io
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

## AGCT	
def hot2seq(hot_array):
  '''Convert the one hot encoding into tokens'''
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

def get_integer_label(y):
  labels = []
  for i in range(len(y)):
    a = y[i]
    labels.append(np.argmax(a))
  yy = np.array(labels)
  return yy

## Stratify Split
stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.99, random_state=42)

## Process Training data
print("Generating Training Data!!")
trainmat = h5py.File('train.mat')
X_train = np.transpose(trainmat['trainxdata'],axes=(2,0,1))
y_train = np.array(trainmat['traindata']).T
yy_train = get_integer_label(y_train)

## get train data in X and y
for train_idx, test_idx in stratSplit.split(X_train, yy_train):
  X = X_train[train_idx]
  y = y_train[train_idx]
  X_temp = X_train[test_idx]
  y_temp = y_train[test_idx]

file_h5 = h5py.File('train.hdf5', 'w')
dt = h5py.special_dtype(vlen=str)
XX = hot2seq(X)
X_h5 = file_h5.create_dataset("X", XX.shape, dt)
y_h5 = file_h5.create_dataset("y", y.shape, 'u1') 
X_h5[0:XX.shape[0]] = XX
y_h5[0:y.shape[0]] = y
file_h5.close()

## get test data in X and y
print("Generating Test Data!!")
yy_train = get_integer_label(y_temp)
for train_idx, test_idx in stratSplit.split(X_temp, yy_train):
  X = X_train[train_idx]
  y = y_train[train_idx]

file_h5 = h5py.File('test.hdf5', 'w')
dt = h5py.special_dtype(vlen=str)
XX = hot2seq(X)
X_h5 = file_h5.create_dataset("X", XX.shape, dt)
y_h5 = file_h5.create_dataset("y", y.shape, 'u1') 
X_h5[0:XX.shape[0]] = XX
y_h5[0:y.shape[0]] = y
file_h5.close()
  
## Process Validation Data -- write npy file
print("Generating Validation Data!!")
validmat = scipy.io.loadmat('valid.mat')
X = np.transpose(validmat['validxdata'],axes=(0,2,1))
y = validmat['validdata']

file_h5 = h5py.File('val.hdf5', 'w')
dt = h5py.special_dtype(vlen=str)
XX = hot2seq(X)
X_h5 = file_h5.create_dataset("X", XX.shape, dt)
y_h5 = file_h5.create_dataset("y", y.shape, 'u1') 
X_h5[0:XX.shape[0]] = XX
y_h5[0:y.shape[0]] = y
file_h5.close()
