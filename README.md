# ChromatinProfilePrediction
Here, we have trained a transformer based model, Longformer, for predicting the Chromatin profile.

# Usage
First, you need to download the dataset for training the model. You can download the preprocessed raw data from: http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz. 

We have to preprocess the data before training/testing the model. 

1. For generating the data for training DeepSea model, please use ```get_train_test_data_OneHot.py``` script. Alternatlively, you can download the pre-processed files from https://1drv.ms/u/s!Aq2K8wPeIwKTgcArx1aHSJ7i3OKduA?e=f41uyC
2. Fore generating the data to train Longformer model. Please use ```get_train_test_data_seq.py``` script. Alternatively, you can download the already processed data from https://1drv.ms/u/s!Aq2K8wPeIwKTgcA8wVmRIKqzHB-50w?e=xHDcC2

# Training Data
Both models have been trained using a subset of training data such that the distribution of labels in the original data remains same in the sampled data. The new train data size comprise of 44,000 samples. 

## Training the model
1. To train the DeepSea model, please run ```ChromatinProfilePrediction/src/DeepSea/train_model.py``` <br/>
   a. Trained model can be downloaded from https://1drv.ms/u/s!Aq2K8wPeIwKTgcAvrAD8NGKZNxSPow?e=DyfXDy
2. To train the longformer model, please run ```ChromatinProfilePrediction/src/train_model.py```<br/>
   a. Trained model can be downloaded using https://1drv.ms/u/s!Aq2K8wPeIwKTgcA_AsZAgQzBQBZZwg?e=7BOHes

## Test the model
1. To obtain the DeepSea model performance, please run ```ChromatinProfilePrediction/src/DeepSea/test_model.py```
2. To test the longformer model, please run ```ChromatinProfilePrediction/src/test_model.py```

# Model Performance
### Using DeepSea
Test Data Loss: 0.0867 <br/>
ROC AUC Score: 0.713 <br/>
Average Precision Score: 0.065 <br/>


### Using Longformer
Test Data Loss: 0.0005 <br/>
ROC AUC Score: 0.608 <br/>
Average Precision Score: 0.035<br/>
