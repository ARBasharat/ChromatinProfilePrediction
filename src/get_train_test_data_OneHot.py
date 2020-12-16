import h5py
import numpy as np
import scipy.io
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

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
X_train = np.transpose(trainmat['trainxdata'],axes=(2,1,0))
y_train = np.transpose(trainmat['traindata'])
yy_train = get_integer_label(y_train)

## get train data in X and y
for train_idx, test_idx in stratSplit.split(X_train, yy_train):
  X = X_train[train_idx]
  y = y_train[train_idx]
  X_temp = X_train[test_idx]
  y_temp = y_train[test_idx]
file_h5 = h5py.File('train_onehot.hdf5', 'w')
dt = h5py.special_dtype(vlen=str)
X_h5 = file_h5.create_dataset("X", X.shape, 'u1')
y_h5 = file_h5.create_dataset("y", y.shape, 'u1') 
X_h5[0:X.shape[0]] = X
y_h5[0:y.shape[0]] = y
file_h5.close()

## get test data in X and y
print("Generating Test Data!!")
yy_train = get_integer_label(y_temp)
for train_idx, test_idx in stratSplit.split(X_temp, yy_train):
  X = X_train[train_idx]
  y = y_train[train_idx]
file_h5 = h5py.File('test_onehot.hdf5', 'w')
dt = h5py.special_dtype(vlen=str)
X_h5 = file_h5.create_dataset("X", X.shape, 'u1')
y_h5 = file_h5.create_dataset("y", y.shape, 'u1') 
X_h5[0:X.shape[0]] = X
y_h5[0:y.shape[0]] = y
file_h5.close()

## Process Validation Data -- write npy file
print("Generating Validation Data!!")
validmat = scipy.io.loadmat('valid.mat')
X = validmat['validxdata']
y = validmat['validdata']
file_h5 = h5py.File('val_onehot.hdf5', 'w')
dt = h5py.special_dtype(vlen=str)
X_h5 = file_h5.create_dataset("X", X.shape, 'u1')
y_h5 = file_h5.create_dataset("y", y.shape, 'u1') 
X_h5[0:X.shape[0]] = X
y_h5[0:y.shape[0]] = y
file_h5.close()

