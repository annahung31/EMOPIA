import sys
import os
import ipdb

from tqdm import tqdm
import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np
from argparse import ArgumentParser


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

import saver

myseed = 42069 
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'



class EmbdDataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 data_path,
                 embd_type,
                 mode):

        self.mode = mode

        self.npz = np.load(os.path.join(data_path, embd_type + '.' + mode + '.npz'))
        self.data = self.npz['x']
        self.target = self.npz['y']
        self.target = [ np.where(aa==1)[0][0] for aa in self.target]
        self.fname = self.npz['fname']
        
        
        self.data = torch.tensor(self.data)
        self.target = torch.tensor(self.target)

        self.seq_len = self.data.shape[1]
        self.dim = self.data.shape[2]


    def __getitem__(self, index):
        if self.mode in ['train', 'val']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


def prep_dataloader(data_path, embd_type, mode, batch_size, n_jobs=0):
    
    dataset = EmbdDataset(data_path, embd_type, mode) 
    
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                          
    return dataloader




class Classifier(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

        # Mean squared error loss
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def cal_loss(self, pred, target):
        
        return self.criterion(pred, target)



def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    learning_rate = config['lr']
    # # Setup optimizer
    # optimizer = getattr(torch.optim, config['optimizer'])(
    #     model.parameters(), **config['optim_hparas'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = 1000.
    loss_record = {'train': [], 'dev': []}     
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                      
        for x, y in tr_set:                
            
            optimizer.zero_grad()              
            x, y = x.to(device), y.to(device)  
            
            pred = model(x)                     
            loss = model.cal_loss(pred, y) 
            loss.backward()                 
            optimizer.step()                    
            loss_record['train'].append(loss.detach().cpu().item())


        dev_mse, acc = dev(dv_set, model, device)
        if dev_mse < min_loss:
           
            min_loss = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f}, acc = {:.4f})'
                .format(epoch + 1, min_loss, acc))
            torch.save(model.state_dict(), config['save_path'])  
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_loss, loss_record


def dev(dv_set, model, device):
    model.eval()                               
    total_loss = 0

    correct = 0
    total = 0
    for x, y in dv_set:                         
        x, y = x.to(device), y.to(device)       
        with torch.no_grad():                  
            pred = model(x)                   
            loss = model.cal_loss(pred, y)  
            _, predicted = torch.max(pred.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

        total_loss += loss.detach().cpu().item() * len(x)  
    total_loss = total_loss / len(dv_set.dataset)            

    return total_loss, correct / total



def test(tt_set, model, device):
    model.eval()                                 
    preds = []
    for x in tt_set:                            
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()    
    return preds


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


if __name__ == "__main__":
    device = get_device()                 
    os.makedirs('models', exist_ok=True)  

    exp_name = 'ignore'
    ROOT = '/home/annahung/189/my_project/emotional-piano/dataset'
    embd_type = 'y_'
    data_path = os.path.join(ROOT, 'embd')
    


    config = {
        'n_epochs': 3000,               
        'batch_size': 8,     
        'lr': 0.0001,         
        # 'optimizer': 'Adam',              
        # 'optim_hparas': {                
        #     'lr': 0.0001,               
        #     'momentum': 0.9             
        # },
        'early_stop': 20,               
        'save_path': 'cls_exps/model_' + exp_name + '.pth' 
    }

    tr_set = prep_dataloader(data_path, embd_type, 'train', config['batch_size'])
    dv_set = prep_dataloader(data_path, embd_type, 'val', config['batch_size'])
    tt_set = prep_dataloader(data_path, embd_type, 'test', config['batch_size'])


    model = Classifier(tr_set.dataset.dim, 128, 3, 4).to(device) 
    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)


    # del model
    # model = Classifier(tr_set.dataset.seq_len, tr_set.dataset.dim).to(device)
    # ckpt = torch.load(config['save_path'], map_location='cpu') 
    # model.load_state_dict(ckpt)

    # preds = test(tt_set, model, device)
    # save_pred(preds, './predict/pred_' + exp_name  + '.csv')         
