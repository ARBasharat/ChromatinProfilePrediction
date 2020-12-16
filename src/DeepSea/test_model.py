import os
import scipy.io
import h5py
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score,roc_curve,auc,average_precision_score,precision_recall_curve
torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark=True

def mkdir(path):
  isExists = os.path.exists(path)
  if not isExists:
    os.makedirs(path)
    print(path+' build successfully!')
    return True
  else:
    print(path+' is already existing!')
    return False

## Hyper Parameters
BATCH_SIZE = 1
print('Load the training data')
file_h5 = h5py.File('test_onehot.hdf5')
testX_data = torch.FloatTensor(file_h5['X'][:])
testY_data = torch.FloatTensor(file_h5['y'][:])
file_h5.close()
print(testX_data.shape, testY_data.shape)
test_loader = Data.DataLoader(dataset=Data.TensorDataset(testX_data, testY_data), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False,)

print('compling the network')
class DeepSEA(nn.Module):
    def __init__(self, ):
        super(DeepSEA, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(53*960, 925)
        self.Linear2 = nn.Linear(925, 919)
    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = x.view(-1, 53*960)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x
      
deepsea = DeepSEA()
deepsea.load_state_dict(torch.load('model/model0526/deepsea_net_params_final.pkl'))
deepsea.cuda()
loss_func = nn.BCEWithLogitsLoss()


print('starting testing')
pred_y = np.zeros([testY_data.shape[0], testY_data.shape[1]])
i=0;j = 0
test_losses = []
deepsea.eval()
for step, (seq, label) in enumerate(test_loader):
    seq = seq.cuda()
    label = label.cuda()
    test_output = deepsea(seq)
    cross_loss = loss_func(test_output, label)
    test_losses.append(cross_loss.item())
    test_output = torch.sigmoid(test_output.cpu().data)     
    pred_y[step, :] = test_output.numpy()[0, :]
    
test_loss = np.average(test_losses)
print_msg = (f'test_loss: {test_loss:.5f}')  
print(print_msg)
# test_loss: 0.08671
# train_loss: 0.08610 valid_loss: 0.08004
mkdir('pred')
np.save('pred/0526pred.npy',pred_y)

print("roc_auc_score:", roc_auc_score(testY_data.data[:], pred_y[:]))
# roc_auc_score: 0.7133197544841413
print("average_precision_score:", average_precision_score(testY_data.data[:], pred_y[:]))
# average_precision_score: 0.06540503333779188


