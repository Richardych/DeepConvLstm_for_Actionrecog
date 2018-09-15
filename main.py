from MyDataset import MyDataset 
import ConvLSTM
import mmd
from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim  
import time
import numpy as np
import pickle as cp
#import theano.tensor as T
from sliding_window import sliding_window
from sklearn.metrics import f1_score

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18
# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24
# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8
# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12
# Batch Size
BATCH_SIZE = 100
# Number filters convolutional layers
NUM_FILTERS = 64
# Size filters convolutional layers
FILTER_SIZE = 5
# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.0007#0.001  from 200epoch
MOMEMTUN = 0.9
L2_WEIGHT = 0.001
EPOCH = 150



def load_dataset(filename):
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()
    X_train, y_train = data[0]
    X_test, y_test = data[1]
    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

def train(model, optimizer, epoch, data_src, data_tar=None):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    batch_j = 0
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
    for batch_id, (data, target) in enumerate(data_src): 
    #for batch_id, batch in enumerate(iterate_minibatches(data_src[0], data_src[1], BATCH_SIZE)):
        _, (x_tar, y_target) = list_tar[batch_j]
        inputs, targets = data.data.view(-1, 1, 24,113).to(DEVICE), target.to(DEVICE).type(torch.cuda.LongTensor)
        if inputs.shape[0] != 100:
            continue
        #x_tar, y_target = x_tar.view(-1, 1, 24,113).to(DEVICE), y_target.to(DEVICE)

        #inputs, targets = batch #(100,24,113)  (100,)
        #inputs,targets = torch.from_numpy(inputs).view(-1, 1, 24,113).to(DEVICE), torch.from_numpy(targets).to(DEVICE).type(torch.cuda.FloatTensor)
        #targets = targets.type(torch.cuda.LongTensor) 

        model.train()
        y_src = model(inputs)
        loss_c = criterion(y_src, targets)
        pred = y_src.data.max(1)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
        loss = loss_c #+ LAMBDA * loss_mmd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train += loss.data
        res_i = 'Epoch: [{}/{}], Batchid: {}, correct:{} ,loss: {:.6f}'.format(epoch,
                EPOCH, batch_id, correct,loss.data)
        tqdm.write(res_i)
    total_loss_train /= batch_id+1
    acc = correct*100.0 / len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, training accuracy: {:.4f}%'.format(
                epoch, EPOCH, total_loss_train, acc)
    with open('train_log.csv','a+') as fff:
        fff.write(res_e+'\n')
    tqdm.write(res_e)
    #from IPython import embed; embed()
    return model

def test(model, epoch, data_src, data_tar=None):
    total_loss_test = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    test_pred = np.empty((0))
    test_true = np.empty((0))
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar)) 
    with torch.no_grad():
        from sklearn.metrics import f1_score
        for batch_id, (data, target) in enumerate(data_src):
             #_, (x_tar, y_target) = list_tar[batch_j]
            inputs, targets = data.data.view(-1, 1, 24,113).to(DEVICE), target.to(DEVICE).type(torch.cuda.LongTensor)
            if inputs.shape[0] != 100:
                continue
            model.eval()
            y_src = model(inputs)
            loss_c = criterion(y_src, targets)
            pred = y_src.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            total_loss_test += loss_c.data
            test_pred = np.append(test_pred, pred, axis=0)
            test_true = np.append(test_true, targets, axis=0)
            res_i = 'Epoch: [{}/{}], Batchid: {}, correct:{} ,loss: {:.6f}'.format(epoch,
                    EPOCH, batch_id, correct, loss_c.data)
            tqdm.write(res_i)
        acc = correct*100.0 / len(data_src.dataset)
        total_loss_test /= (batch_id+1)
        f1 = f1_score(test_true, test_pred, average='weighted')
        res_e = 'Epoch: [{}/{}], test loss: {:.6f}, test accuracy: {:.4f}%, f1:{}'.format(
            epoch, EPOCH, total_loss_test,  acc, f1)
        tqdm.write(res_e)


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


#(557963,113)  (118750,113)
X_train, y_train, X_test, y_test = load_dataset('oppChallenge_gestures.data')
assert NB_SENSOR_CHANNELS == X_train.shape[1]
X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)  #(9894, 24, 113) 

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(MyDataset(X_train, y_train), 
                        batch_size=BATCH_SIZE, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(MyDataset(X_test, y_test), 
                        batch_size=BATCH_SIZE, shuffle=False, **kwargs)

model = ConvLSTM.ConvLSTM(num_classes=18)
model = torch.load('model.pkl')

model = model.to(DEVICE)
optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMEMTUN,
        weight_decay=L2_WEIGHT
        )
print(model)

#train
for e in tqdm(range(1, EPOCH + 1)):
    model = train(model=model, optimizer=optimizer,
            #epoch=e, data_src=[X_train,y_train], data_tar=[X_test,y_test])
            epoch=e, data_src=train_loader, data_tar = train_loader)
    if e%2 == 0:
        torch.save(model, 'model.pkl')

"""

#test
for e in tqdm(range(1,1 + 1)):
    test(model, e, data_src=test_loader, data_tar=test_loader)

"""











