import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class ConvLSTM(nn.Module):
    """
    def __init__(self, n_input=28 * 28, n_hidden=256, n_class=18):
        super(ConvLSTM, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(n_hidden, n_class)
    
    def forward(self, src, tar):
        x_src = self.layer_input(src)
        x_tar = self.layer_input(tar)
        x_src = self.dropout(x_src)
        x_tar = self.dropout(x_tar)
        x_src_mmd = self.relu(x_src)
        x_tar_mmd = self.relu(x_tar)
        x_src = self.layer_hidden(x_src_mmd)
        return x_src, x_src_mmd, x_tar_mmd
    """


    def __init__(self, num_classes = 18):
        super(ConvLSTM,self).__init__()
        #batchsize,1,24,113
        self.NUM_FILTERS = 128
        self.BATCH = 100
        self.conv1 = nn.Conv2d(1,64,5,(1,1))
        self.conv2 = nn.Conv2d(64,128,5,(1,1))
        self.conv3 = nn.Conv2d(128,256,5,(1,1))
        self.conv4 = nn.Conv2d(256,512,5,(1,1))
        self.dropout = nn.Dropout(0.5)
        #self.shuff = a.permute(,,,,)
        #inpdim, hiden units, lstm layers,
        #self.lstm1 = nn.LSTM(8, self.NUM_FILTERS, 2,batch_first=True)
        self.lstm1 = nn.LSTM(512, self.NUM_FILTERS, 1,batch_first=True)
        self.lstm2 = nn.LSTM(128,128,1,batch_first=True)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        convs = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        #convs = convs.view(self.BATCH, 64*97, 8)
        convs = convs.view(self.BATCH, 8*97, 512)
        lstm1,_ = self.lstm1(self.dropout(convs))
        lstm2,_ = self.lstm2(self.dropout(lstm1))
        #lstm2 = lstm2[:,-1,:]
        lstm2 = lstm2.contiguous().view(-1,128)
        fc = self.fc(lstm2)
        fc2 = fc.view(self.BATCH, -1, 18)[:,-1,:]
        return fc2

"""
x = torch.rand(64,1,24,113)
model = ConvLSTM()
res = model(x)
print(res.size())
#from IPython import embed; embed()
"""



















