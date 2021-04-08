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
import collections
import pandas as pd
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
                 task_type,
                 mode, 
                 data_mode):

        if task_type not in ['4-cls', 'Arousal', 'Valence']:
            print("invalid task type, only accept '4-cls', 'Arousal', 'Valence'")
            return None
        

        self.mode = mode

        self.npz = np.load(os.path.join(data_path, embd_type + '.' + mode + '.npz'))
        self.data = self.npz['x']

        if data_mode == 'Joanna':
            self.target = self.npz['y']
            self.target = [ np.where(aa==1)[0][0] for aa in self.target]
            print('origin label:    ', self.target[-5:])
            if task_type == 'Arousal':
                self.target = [0 if x in [0,1] else 1 for x in self.target]
            elif task_type == 'Valence':
                self.target = [0 if x in [0,3] else 1 for x in self.target]

            print('transfered label:', self.target[-5:])

            self.target = torch.tensor(self.target)

        self.fname = self.npz['fname']
        
        self.data = torch.tensor(self.data)
        

        self.seq_len = self.data.shape[1]
        self.dim = self.data.shape[2]

        print(mode , ':', self.data.shape)

    def __getitem__(self, index):
        
        if data_mode == 'Joanna' and self.mode != 'test':
            return self.data[index], self.target[index]
        
        elif data_mode == 'Joanna' and self.mode == 'test':
            return self.data[index], self.target[index], self.fname[index]

        else:
            return self.data[index], self.fname[index]


    def __len__(self):
        return len(self.data)


def prep_dataloader(data_path, embd_type, task_type, mode, batch_size, n_jobs=0, data_mode='Joanna'):
    
    dataset = EmbdDataset(data_path, embd_type, task_type,  mode, data_mode) 
    
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                          
    return dataloader




class Classifier(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, device, input_size, hidden_size, num_layers, num_classes):
        super(Classifier, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

        # Mean squared error loss
        self.criterion = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

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

    max_acc = 0.
    loss_record = {'train': [], 'dev': [], 'test': 0}     
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()         
        total_loss = 0             
        for x, y in tr_set:                
            
            optimizer.zero_grad()              
            x, y = x[:, 1:, :].to(device), y.to(device)  
            # print(x.shape)
            # ipdb.set_trace()
            pred = model(x)                    
            ipdb.set_trace() 
            loss = model.cal_loss(pred, y) 
            total_loss += loss.detach().cpu().item() * len(x)
            loss.backward()                 
            optimizer.step()                    
        
        total_loss = total_loss / len(tr_set.dataset)
        loss_record['train'].append(total_loss)


        dev_mse, acc = dev(dv_set, model, device)
        if acc > max_acc:
            max_acc = acc
            print('Saving model (epoch = {:4d}, loss = {:.4f}, acc = {:.4f})'
                .format(epoch + 1, dev_mse, acc))
            torch.save(model.state_dict(), os.path.join(config['save_path'], 'model.pth'))  
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        epoch += 1
        loss_record['dev'].append([dev_mse, acc])
        
        
        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return max_acc, loss_record


def dev(dv_set, model, device):
    model.eval()                               
    total_loss = 0

    correct = 0
    total = 0
    for x, y in dv_set:                         
        x, y = x[:, 1:, :].to(device), y.to(device)       
        with torch.no_grad():                  
            pred = model(x)                   
            loss = model.cal_loss(pred, y)  
            _, predicted = torch.max(pred.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

        total_loss += loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)            

    return total_loss, correct / total


def test_inference(tt_set, model, device):
    model.eval()                                 

    correct = 0
    total = 0
    results = []
    files = []
    for x, fname in tt_set:                            
        x = x.to(device)             
        with torch.no_grad():                   
            pred = model(x)                     
            _, predicted = torch.max(pred.data, 1)
            results.append(predicted.item())
            files.append(fname)
            print('predict: {:.4f}/{:.4f}, result: {}, name: {}'.format(pred[0][0], pred[0][1], predicted.item(), fname))
            
    print(collections.Counter(results))

    df = pd.DataFrame({'filename': files, 'predict': results})
    df.to_csv('predict_result.csv')

def test(tt_set, model, exp_name, device):
    model.eval()                                 

    correct = 0
    total = 0
    results = []
    files = []
    for x, y, fname in tt_set:     
        
        x, y = x[:, 1:, :].to(device), y.to(device)         
        
        with torch.no_grad():                   
            pred = model(x)                     
            _, predicted = torch.max(pred.data, 1)

            results.append(predicted.item())
            files.append(fname[0].split('/')[-1][:-4])

            total += y.size(0)
            correct += (predicted == y).sum().item()
            # print('predict: {:.4f}/{:.4f}, target: {}'.format(pred[0][0], pred[0][1], y))
            
    df = pd.DataFrame({'filename': files, 'predict': results})
    # df.to_csv('predict_result_' + exp_name + '.csv')

    # acc = correct / total
    acc = correct / len(tt_set.dataset)
    print('test acc: {:.4f}'.format(acc))

    return acc



if __name__ == "__main__":
    device = get_device()                 
    os.makedirs('models', exist_ok=True)  

    
    ROOT = '/home/annahung/189/my_project/emotional-piano/dataset'
    task_type = 'Valence'   # ['4-cls', 'Arousal', 'Valence']
    print('task type: ', task_type)
    embd_type = 'layer_8.y_'
    data_path = os.path.join(ROOT, 'embd', 'split')

    mode = 'train'
    data_mode = 'Joanna'   # 'Joanna', 'other'
    
    exp_name = 'ignore'
    print('exp:', exp_name)
    config = {
        'n_epochs': 3000,               
        'batch_size': 4,     
        'lr': 0.0001,         
        # 'optimizer': 'Adam',              
        # 'optim_hparas': {                
        #     'lr': 0.0001,               
        #     'momentum': 0.9             
        # },
        'early_stop': 10,               
        'save_path': 'cls_exps/' + exp_name
    }
    
    if data_mode == 'Joanna':
        tr_set = prep_dataloader(data_path, embd_type, task_type, 'train', config['batch_size'])
        dv_set = prep_dataloader(data_path, embd_type, task_type, 'val', config['batch_size'])
        tt_set = prep_dataloader(data_path, embd_type, task_type, 'test',1)
    
    else:
        tt_set = prep_dataloader(data_path, embd_type, task_type, 'test',1, data_mode=data_mode)

    if task_type == '4-cls':
        num_classes = 4
    else:
        num_classes = 2

    print('num of classes:', num_classes)
    model = Classifier(device, tt_set.dataset.dim, 128, 4, num_classes).to(device)
    if mode == 'train':
        os.makedirs(config['save_path'], exist_ok=True)
        model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

        # testing
        ckpt = torch.load(os.path.join(config['save_path'], 'model.pth'))  
        model.load_state_dict(ckpt)
        test_acc = test(tt_set, model, device)

        model_loss_record['test'] = test_acc

        with open(os.path.join(config['save_path'], 'loss.json'), 'w') as f:
            json.dump(model_loss_record, f)

        print('result saved to', config['save_path'])
        

    if mode == 'test':
        
        ckpt = torch.load(os.path.join(config['save_path'], 'model.pth'))  # Load your best model
        model.load_state_dict(ckpt)
        if data_mode == 'Joanna':
            test_acc = test(tt_set, model, exp_name, device)
        else:
            test_inference(tt_set, model, device)
