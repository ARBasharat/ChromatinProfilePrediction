# ChromatinProfilePrediction
Here, we have trained a transformer based model, Longformer, for predicting the Chromatin profile.

# Usage
First, you need to download the dataset for training the model. You can download the preprocessed raw data from: http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz. 

We have to preprocess the data before training/testing the model. 

1. For generating the data for training DeepSea model, please use ```get_train_test_data_OneHot.py``` script. Alternatlively, you can download the pre-processed files from https://1drv.ms/u/s!Aq2K8wPeIwKTgcArx1aHSJ7i3OKduA?e=f41uyC
2. Fore generating the data to train Longformer model. Please use ```get_train_test_data_seq.py``` script. Alternatively, you can download the already processed data from https://1drv.ms/u/s!Aq2K8wPeIwKTgcA8wVmRIKqzHB-50w?e=xHDcC2

## Training the model
1. To train the DeepSea model, please run ```ChromatinProfilePrediction/src/DeepSea/train_model.py```
2. To train the longformer model, please run ```ChromatinProfilePrediction/src/train_model.py```

## Test the model
1. To obtain the DeepSea model performance, please run ```ChromatinProfilePrediction/src/DeepSea/test_model.py```
2. To test the longformer model, please run ```ChromatinProfilePrediction/src/test_model.py```

# Model Performance
### Using DeepSea
Test Data Loss: 0.0867 <br/>
ROC AUC Score: 0.713 <br/>
Average Precision Score: 0.065 <br/>


### Using Longformer
Test Data Loss:  <br/>
ROC AUC Score: <br/>
Average Precision Score: <br/>
