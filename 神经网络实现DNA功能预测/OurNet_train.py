import scipy.io
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import os
import time
torch.cuda.set_device(3)
torch.manual_seed(1337)
np.random.seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cudnn.benchmark=True

## Hyper Parameters
EPOCH = 60
BATCH_SIZE = 100
LR = 0.001
save_model_time = '1220'

mkpath = 'model/model%s'% save_model_time

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+' build successfully!')
        return True
    else:
        print(path+' is already existing!')
        return False
        
def bestmodel(net,save_model_time,valid_loss):
    bestloss = 10000
    if valid_loss < bestloss :
        bestloss = valid_loss
        torch.save(net, 'model/model{save_model_time}/ournet_net_bestmodel.pkl'.format(save_model_time=save_model_time))
        torch.save(net.state_dict(), 'model/model{save_model_time}/ournet_net_params_bestmodel.pkl'.format(save_model_time=save_model_time))
    return True     

mkdir(mkpath)
print('starting loading the data')
np_valid_data = scipy.io.loadmat('valid.mat')

validX_data = torch.FloatTensor(np_valid_data['validxdata'])
validY_data = torch.FloatTensor(np_valid_data['validdata'])

params = {'batch_size': 100,'num_workers': 4}

valid_loader = Data.DataLoader(
    dataset=Data.TensorDataset(validX_data, validY_data), 
    shuffle=False,
    **params)

print('compling the network')
class OurNet(nn.Module):
    def __init__(self, ):
        super(OurNet, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(75*640, 925)
        self.Linear2 = nn.Linear(925, 919)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n,h_c) = self.BiLSTM(x_x)
        #x, h_n = self.BiGRU(x_x)
        x = x.contiguous().view(-1, 75*640)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x


ournet = OurNet()
ournet.cuda()
print(ournet)

optimizer = optim.RMSprop(ournet.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,verbose=1)
loss_func = nn.BCEWithLogitsLoss()

print('starting training')
# training and validating
since = time.time()

train_losses = []
valid_losses = []

for epoch in range(EPOCH):
    ournet.train()
    train_loss = 0
    for i in range(1,11):
        trainX_data = torch.load('pt_data/%s.pt' % str(i))
        trainY_data = torch.load('pt_label/%s.pt' % str(i))
        train_loader = Data.DataLoader(dataset=Data.TensorDataset(trainX_data, trainY_data), shuffle=True, **params)
        for step, (train_batch_x, train_batch_y) in enumerate(train_loader):

            train_batch_x = train_batch_x.cuda()
            train_batch_y = train_batch_y.cuda()

            out = ournet(train_batch_x)
            loss = loss_func(out, train_batch_y)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
    i = 1

    if epoch % 5 == 0:
       torch.save(ournet, 'model/model{save_model_time}/ournet_net_{epoch}.pkl'.format(save_model_time=save_model_time,epoch=int(epoch/5)))
       torch.save(ournet.state_dict(), 'model/model{save_model_time}/ournet_net_params_{epoch}.pkl'.format(save_model_time=save_model_time,epoch=int(epoch/5)))

    
    ournet.eval()

    for valid_step, (valid_batch_x, valid_batch_y) in enumerate(valid_loader):

        valid_batch_x = valid_batch_x.cuda()
        valid_batch_y = valid_batch_y.cuda()

        val_out = ournet(valid_batch_x)
        val_loss = loss_func(val_out, valid_batch_y)
        valid_losses.append(val_loss.item())
        
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    
    scheduler.step(valid_loss)
    
    epoch_len = len(str(epoch))

    print_msg = (f'[{epoch:>{epoch_len}}/{EPOCH:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)

    #save bestmodel
    bestmodel(ournet,save_model_time,valid_loss)
    
    train_losses = []
    valid_losses = []

time_elapsed = time.time() - since
print('time:', time_elapsed)
torch.save(danq, 'model/model{save_model_time}/ournet_net_final.pkl'.format(save_model_time=save_model_time))  # save entire net
torch.save(danq.state_dict(), 'model/model{save_model_time}/ournet_net_params_final.pkl'.format(save_model_time=save_model_time))

print('starting loading the data')
np_test_data = scipy.io.loadmat('test.mat')
testX_data = torch.FloatTensor(np_test_data['testxdata'])
testY_data = torch.FloatTensor(np_test_data['testdata'])

test_loader = Data.DataLoader(
   dataset=Data.TensorDataset(testX_data, testY_data),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
   drop_last=False,
)
ournet.load_state_dict(torch.load('model/model{save_model_time}/ournet_net_params_final.pkl'.format(save_model_time=save_model_time)))
ournet.cuda()

print('starting testing')
# training
pred_y = np.zeros([455024, 919])
i=0;j = 0
test_losses = []
ournet.eval()
for step, (seq, label) in enumerate(test_loader):
    #print(step)
    seq = seq.cuda()
    label = label.cuda()

    test_output = ournet(seq)
    cross_loss = loss_func(test_output, label)
    test_losses.append(cross_loss.item())
    
    test_output = torch.sigmoid(test_output.cpu().data)     

    if(step<4550):
        for i, j in zip(range(step*100, (step+1)*100),range(0, 100)):
            pred_y[i, :] = test_output.numpy()[j, :]
    else:
        for i,j in zip(range(455000,455024),range(0,24)):
            pred_y[i, :] = test_output.numpy()[j, :]
        #print(test_output.numpy())
        
    
test_loss = np.average(test_losses)
print_msg = (f'test_loss: {test_loss:.5f}')  
print(print_msg)    
np.save('pred.npy',pred_y)
