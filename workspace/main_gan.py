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

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


import saver

from models import TransformerModel, network_paras
from utils import write_midi


myseed = 42069
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():    
    torch.cuda.manual_seed_all(myseed)


################################################################################
# config
################################################################################

parser = ArgumentParser()
parser.add_argument("--mode",default="train",type=str,choices=["train", "inference", "embd"])
parser.add_argument("--task_type",default="Arousal",type=str,choices=['4-cls', 'Arousal', 'Valence', 'Arousal_1', 'Arousal_2'])
parser.add_argument("--path_train_data", default='joanna_noBroken_train', type=str)   #joanna_noBroken_train
parser.add_argument("--gid", default= 0, type=int)


# training config
### python main-cp.py --mode train --load_ckt 0309-1857 --load_ckt_loss 50 --exp_name 0315-1234

parser.add_argument("--exp_name", default='0417-1149' , type=str)
parser.add_argument("--load_ckt", default="0309-1857", type=str)   #pre-train model
parser.add_argument("--load_ckt_loss", default="30", type=str)     #pre-train model

# inference config
### python main-cp.py --mode inference --load_ckt 0309-1857 --load_ckt_loss 50 --num_songs 5 --emo_tag 1

parser.add_argument("--num_songs", default=5, type=int)
parser.add_argument("--emo_tag", default=1, type=int)

args = parser.parse_args()


MODE = args.mode
task_type = args.task_type

###--- data ---###
path_data_root = '../dataset/co-representation/'
# path_data_root = '../dataset_wayne/representations/'

path_train_data = os.path.join(path_data_root, 'no_val', args.path_train_data + '_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, 'more_dictionary.pkl')
path_train_idx = os.path.join(path_data_root, 'no_val', args.path_train_data + '_fn2idx_map.json')
path_train_data_cls_idx = os.path.join(path_data_root, 'no_val', args.path_train_data + '_data_idx.npz')


###--- training config ---###
 
if MODE == 'train':
    path_exp = 'exp/' + args.exp_name

batch_size = 4      #4
gid = args.gid
init_lr = 0.00001   #0.0001


###--- fine-tuning & inference config ---###
#info_load_model = None
info_load_model = (
      # path to ckpt for loading
      'exp/' + args.load_ckt,
      # loss
      args.load_ckt_loss                               
      )

path_gendir = os.path.join('exp/' + args.load_ckt, 'gen_midis', 'loss_'+ args.load_ckt_loss)
num_songs = args.num_songs
emotion_tag = args.emo_tag

################################################################################
# File IO
################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)


##########################################################################################################################
# Script
##########################################################################################################################

def take_embd(path_train_idx):

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)
    
    with open(path_train_idx, 'r') as f:
        idx_info = json.load(f)

    embd_root = '../dataset/embd'

    # config
    n_class = []  # number of classes of each token. [56, 127, 18, 4, 85, 18, 41, 5]
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init
    net = TransformerModel(n_class)
    net.cuda()
    net.eval()

    # load model
    path_ckpt = 'exp/0309-1857' # path to ckpt dir
    loss = 30 # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
    print('[*] load model from:',  path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))


    # unpack
    batch_size = 1
    train_x = train_data['x']
    train_y = train_data['y']

    num_batch = len(train_x) // batch_size
    idx2fn = idx_info['idx2fn']
    
    # embd_list = []
    for bidx in tqdm(range(num_batch)): # num_batch 
        # index
        bidx_st = batch_size * bidx
        bidx_ed = batch_size * (bidx + 1)

        # unpack batch data
        filename = idx2fn[str(bidx_st)]
        batch_x = train_x[bidx_st:bidx_ed]
        batch_y = train_y[bidx_st:bidx_ed]

        # to tensor
        batch_x = torch.from_numpy(batch_x).long().cuda()
        batch_y = torch.from_numpy(batch_y).long().cuda()
        
        # net.train_step(batch_x, batch_y, batch_mask)
        h, y_type, layer_outputs  = net.forward_hidden(batch_x)
    
        # _, _, _, _, _, _, _, y_ = net.forward_output(h, batch_y)

        # _, _, _, _, _, _, _, four_y_ = net.forward_output(layer_outputs[3], batch_y)
        _, _, _, _, _, _, _, eight_y_ = net.forward_output(layer_outputs[7], batch_y)

        # np.save(os.path.join(embd_root, 'y_', filename + '.npy'), y_.detach().cpu().numpy())
        # np.save(os.path.join(embd_root, 'h', filename + '.npy'), h.detach().cpu().numpy())
        # np.save(os.path.join(embd_root, 'layer_4.y_', filename + '.npy'), four_y_.detach().cpu().numpy())
        np.save(os.path.join(embd_root, 'layer_8.y_.local', filename + '.npy'), eight_y_.detach().cpu().numpy())
        


class PEmoDataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 
                 task_type):

        
        
        self.train_data = np.load(path_train_data)
        self.train_x = self.train_data['x']
        self.train_y = self.train_data['y']
        self.train_mask = self.train_data['mask']
        
        self.cls_idx = np.load(path_train_data_cls_idx)
        self.cls_1_idx = self.cls_idx['cls_1_idx']
        self.cls_2_idx = self.cls_idx['cls_2_idx']
        self.cls_3_idx = self.cls_idx['cls_3_idx']
        self.cls_4_idx = self.cls_idx['cls_4_idx']
        
        if task_type == 'Arousal':
            print('preparing data for training "Arousal"')
            self.label_transfer('Arousal')

        elif task_type == 'Arousal_1':
            #only train on one class
            self.label_transfer('Arousal')
            self.train_x = self.train_x[list(self.cls_1_idx) + list(self.cls_2_idx)]
        
        elif task_type == 'Arousal_2':
            self.label_transfer('Arousal')
            self.train_x = self.train_x[list(self.cls_3_idx) + list(self.cls_4_idx)]  
                
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


def train_cls():

    n_epoch = 4000
    max_grad_norm = 3
    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    
    train_loader = prep_dataloader(args.task_type, batch_size)

    # create saver
    saver_agent = saver.Saver(path_exp)
    



def train():
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
    n_class = []  # number of classes of each token. [56, 127, 18, 4, 85, 18, 41, 5]
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))
    
    # log
    print('num of classes:', n_class)
    
    # init
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
        net.load_state_dict(torch.load(path_saved_ckpt))

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # unpack


    
    # print('    train_x:', train_x.shape)
    # print('    train_y:', train_y.shape)
    # print('    train_mask:', train_mask.shape)

    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(8)


        num_batch = len(train_loader)
        print('    num_batch:', num_batch)

        for bidx, (batch_x, batch_y, batch_mask)  in enumerate(train_loader): # num_batch 
            saver_agent.global_step_increment()

            # to tensor
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_mask = batch_mask.cuda()

            # run
            losses = net.train_step(batch_x, batch_y, batch_mask)

            final_res = net.inference_during_training(dictionary, emotion_tag=1)
            
            
            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6] + losses[7]) / 8
         

            # Update
            net.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # print
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
        elif 0.05 < loss <= 0.40:
            fn = int(loss * 100)
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif loss <= 0.05:
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

    # init model
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()
    
    # load model
    print('[*] load model from:',  path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))
    
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
        path_outfile = os.path.join(path_gendir, 'emo_{}_{}'.format( str(emotion_tag), str(sidx)))
            
        res = net.inference_from_scratch(dictionary, emotion_tag)
        np.save(path_outfile + '.npy', res)
        
        write_midi(res, path_outfile + '.mid', word2event)

        song_time = time.time() - start_time
        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

        sidx += 1
        # except KeyboardInterrupt:
        #     raise ValueError(' [x] terminated.')
        # except:
        #     # continue
        #     exit()
  
    
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
    elif MODE == 'embd':
        take_embd(path_train_idx)
    else:
        pass
