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
from collections import OrderedDict
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


import saver
from models import TransformerModel, network_paras
from utils import write_midi, get_random_string


################################################################################
# config
################################################################################

parser = ArgumentParser()
parser.add_argument("--mode",default="train",type=str,choices=["train", "inference"])
parser.add_argument("--task_type",default="4-cls",type=str,choices=['4-cls', 'Arousal', 'Valence', 'ignore'])
parser.add_argument("--gid", default= 0, type=int)
parser.add_argument("--data_parallel", default= 0, type=int)

parser.add_argument("--exp_name", default='output' , type=str)
parser.add_argument("--load_ckt", default="none", type=str)   #pre-train model
parser.add_argument("--load_ckt_loss", default="25", type=str)     #pre-train model
parser.add_argument("--path_train_data", default='emopia', type=str)  
parser.add_argument("--data_root", default='../dataset/co-representation/', type=str)
parser.add_argument("--load_dict", default="more_dictionary.pkl", type=str)
parser.add_argument("--init_lr", default= 0.00001, type=float)
# inference config

parser.add_argument("--num_songs", default=5, type=int)
parser.add_argument("--emo_tag", default=1, type=int)
parser.add_argument("--out_dir", default='none', type=str)
args = parser.parse_args()

print('=== args ===')
for arg in args.__dict__:
    print(arg, args.__dict__[arg])
print('=== args ===')
# time.sleep(10)    #sleep to check again if args are right


MODE = args.mode
task_type = args.task_type


###--- data ---###
path_data_root = args.data_root

path_train_data = os.path.join(path_data_root, args.path_train_data + '_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, args.load_dict)
path_train_idx = os.path.join(path_data_root, args.path_train_data + '_fn2idx_map.json')
path_train_data_cls_idx = os.path.join(path_data_root, args.path_train_data + '_data_idx.npz')

assert os.path.exists(path_train_data)
assert os.path.exists(path_dictionary)
assert os.path.exists(path_train_idx)

# if the dataset has the emotion label, get the cls_idx for the dataloader
if args.path_train_data == 'emopia':    
    assert os.path.exists(path_train_data_cls_idx)

###--- training config ---###
 
if MODE == 'train':
    path_exp = 'exp/' + args.exp_name

if args.data_parallel > 0:
    batch_size = 8
else:
    batch_size = 4      #4

gid = args.gid
init_lr = args.init_lr   #0.0001


###--- fine-tuning & inference config ---###
if args.load_ckt == 'none':
    info_load_model = None
    print('NO pre-trained model used')

else:
    info_load_model = (
        # path to ckpt for loading
        'exp/' + args.load_ckt,
        # loss
        args.load_ckt_loss                               
        )


if args.out_dir == 'none':
    path_gendir = os.path.join('exp/' + args.load_ckt, 'gen_midis', 'loss_'+ args.load_ckt_loss)
else:
    path_gendir = args.out_dir

num_songs = args.num_songs
emotion_tag = args.emo_tag


################################################################################
# File IO
################################################################################

if args.data_parallel == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)


##########################################################################################################################
# Script
##########################################################################################################################


class PEmoDataset(Dataset):
    def __init__(self,
                 
                 task_type):

        self.train_data = np.load(path_train_data)
        self.train_x = self.train_data['x']
        self.train_y = self.train_data['y']
        self.train_mask = self.train_data['mask']
        
        if task_type != 'ignore':
        
            self.cls_idx = np.load(path_train_data_cls_idx)
            self.cls_1_idx = self.cls_idx['cls_1_idx']
            self.cls_2_idx = self.cls_idx['cls_2_idx']
            self.cls_3_idx = self.cls_idx['cls_3_idx']
            self.cls_4_idx = self.cls_idx['cls_4_idx']
        
            if task_type == 'Arousal':
                print('preparing data for training "Arousal"')
                self.label_transfer('Arousal')

            elif task_type == 'Valence':
                print('preparing data for training "Valence"')
                self.label_transfer('Valence')


        self.train_x = torch.from_numpy(self.train_x).long()
        self.train_y = torch.from_numpy(self.train_y).long()
        self.train_mask = torch.from_numpy(self.train_mask).float()


        self.seq_len = self.train_x.shape[1]
        self.dim = self.train_x.shape[2]
        
        print('train_x: ', self.train_x.shape)

    def label_transfer(self, TYPE):
        if TYPE == 'Arousal':
            for i in range(self.train_x.shape[0]):
                if self.train_x[i][0][-1] in [1,2]:
                    self.train_x[i][0][-1] = 1
                elif self.train_x[i][0][-1] in [3,4]:
                    self.train_x[i][0][-1] = 2
        
        elif TYPE == 'Valence':
            for i in range(self.train_x.shape[0]):
                if self.train_x[i][0][-1] in [1,4]:
                    self.train_x[i][0][-1] = 1
                elif self.train_x[i][0][-1] in [2,3]:
                    self.train_x[i][0][-1] = 2   

        
        
    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index], self.train_mask[index]
    

    def __len__(self):
        return len(self.train_x)


def prep_dataloader(task_type, batch_size, n_jobs=0):
    
    dataset = PEmoDataset(task_type) 
    
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=False, drop_last=False,
        num_workers=n_jobs, pin_memory=True)                          
    return dataloader




def train():

    myseed = 42069
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():    
        torch.cuda.manual_seed_all(myseed)


    # hyper params
    n_epoch = 4000
    max_grad_norm = 3

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    
    train_loader = prep_dataloader(args.task_type, batch_size)

    # create saver
    saver_agent = saver.Saver(path_exp)

    # config
    n_class = []  # number of classes of each token. [56, 127, 18, 4, 85, 18, 41, 5]  with key: [... , 25]
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))
    
    

    n_token = len(n_class)
    # log
    print('num of classes:', n_class)
    
    # init
    

    if args.data_parallel > 0 and torch.cuda.count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = TransformerModel(n_class, data_parallel=True)
        net = nn.DataParallel(net)

    else:
        net = TransformerModel(n_class)

    net.cuda()
    net.train()
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    
    # load model
    if info_load_model:
        path_ckpt = info_load_model[0] # path to ckpt dir
        loss = info_load_model[1] # loss
        name = 'loss_' + str(loss)
        path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
        print('[*] load model from:',  path_saved_ckpt)
        
        try:
            net.load_state_dict(torch.load(path_saved_ckpt))
        except:
            # print('WARNING!!!!! Not the whole pre-train model is loaded, only load partial')
            # print('WARNING!!!!! Not the whole pre-train model is loaded, only load partial')
            # print('WARNING!!!!! Not the whole pre-train model is loaded, only load partial')
            # net.load_state_dict(torch.load(path_saved_ckpt), strict=False)
        
            state_dict = torch.load(path_saved_ckpt)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name] = v
            
            net.load_state_dict(new_state_dict)


    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)


    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(n_token)


        num_batch = len(train_loader)
        print('    num_batch:', num_batch)

        for bidx, (batch_x, batch_y, batch_mask)  in enumerate(train_loader): # num_batch 
            saver_agent.global_step_increment()

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_mask = batch_mask.cuda()
            
            
            losses = net(batch_x, batch_y, batch_mask)

            if args.data_parallel > 0:
                loss = 0
                calculated_loss = []
                for i in range(n_token):
                    
                    loss += ((losses[i][0][0] + losses[i][0][1]) / (losses[i][1][0] + losses[i][1][1]))
                    calculated_loss.append((losses[i][0][0] + losses[i][0][1]) / (losses[i][1][0] + losses[i][1][1]))
                loss = loss / n_token
                
                
            else:
                loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6] + losses[7]) / 8


            # Update
            net.zero_grad()
            loss.backward()
                
                
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            if args.data_parallel > 0:
                
                sys.stdout.write('{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                        bidx, num_batch, loss, calculated_loss[0], calculated_loss[1], calculated_loss[2], calculated_loss[3], calculated_loss[4], calculated_loss[5], calculated_loss[6], calculated_loss[7]))
                sys.stdout.flush()


                # acc
                acc_losses += np.array([l.item() for l in calculated_loss])



            else:
                
                sys.stdout.write('{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                        bidx, num_batch, loss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6], losses[7]))
                sys.stdout.flush()


                # acc
                acc_losses += np.array([l.item() for l in losses])



            acc_loss += loss.item()

            # log
            saver_agent.add_summary('batch loss', loss.item())

        
        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        print('------------------------------------')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        
        each_loss_str = '{:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                  acc_losses[0], acc_losses[1], acc_losses[2], acc_losses[3], acc_losses[4], acc_losses[5], acc_losses[6], acc_losses[7])
        
        print('    >', each_loss_str)

        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)

        # save model, with policy
        loss = epoch_loss
        if 0.4 < loss <= 0.8:
            fn = int(loss * 10) * 10
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif 0.08 < loss <= 0.40:
            fn = int(loss * 100)
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif loss <= 0.08:
            print('Finished')
            return  
        else:
            saver_agent.save_model(net, name='loss_high')


def generate():

    # path
    path_ckpt = info_load_model[0] # path to ckpt dir
    loss = info_load_model[1] # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir
    os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []   # num of classes for each token
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))


    n_token = len(n_class)

    # init model
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()
    
    # load model
    print('[*] load model from:',  path_saved_ckpt)
    
    
    try:
        net.load_state_dict(torch.load(path_saved_ckpt))
    except:
        state_dict = torch.load(path_saved_ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
            
        net.load_state_dict(new_state_dict)


    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0 
    sidx = 0
    while sidx < num_songs:
        # try:
        start_time = time.time()
        print('current idx:', sidx)

        if n_token == 8:
            path_outfile = os.path.join(path_gendir, 'emo_{}_{}'.format( str(emotion_tag), get_random_string(10)))        
            res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token)
        

        if res is None:
            continue
        np.save(path_outfile + '.npy', res)
        write_midi(res, path_outfile + '.mid', word2event)

        song_time = time.time() - start_time
        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

        sidx += 1

    
    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time':song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)





if __name__ == '__main__':
    # -- training -- #
    if MODE == 'train':
        train()
    # -- inference -- #
    elif MODE == 'inference':
        generate()
    else:
        pass
