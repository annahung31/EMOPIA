
import os
import glob
import random
import string
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def write_midi(words, path_outfile, word2event):
    
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]
                
                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)
                
                all_notes.append(
                    Note(
                        pitch=int(pitch), 
                        start=cur_pos, 
                        end=end, 
                        velocity=int(velocity))
                    )
            except:
                continue
        else:
            pass
    
    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)


################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    if np.isnan(probs).any():
        return None
    else:
        return probs



## gumbel
def gumbel_softmax(logits, temperature):
    return F.gumbel_softmax(logits, tau=temperature, hard=True)


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    try:
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    except:
        ipdb.set_trace()
    return word



def sampling(logit, p=None, t=1.0, is_training=False):
    

    if is_training:
        logit = logit.squeeze()
        probs = gumbel_softmax(logits=logit, temperature=t)
        
        return torch.argmax(probs)
        
    else:
        logit = logit.squeeze().cpu().numpy()
        probs = softmax_with_temperature(logits=logit, temperature=t)
    
        if probs is None:
            return None

        if p is not None:
            cur_word = nucleus(probs, p=p)
            
        else:
            cur_word = weighted_sampling(probs)
        return cur_word






def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str



'''
假如 classifier 是 pre-trained, 
那就要過 gumbel softmax (因為 classifer 是看 real data)
假如不是，classsifier 直接吃 training phase 的東西，那直接喂 logit/ softmax 也可以

KEY: classifier 有沒有要看 real data
有： gumbel
沒有： 不用 gumbel

喂給 classifier 有 2 種選擇：
1. logit
2. probs -> 不行，因為還是得要在喂給 forward_hidden... 所以需要  word
'''

'''
def compile_data(test_folder):
    MAX_LEN = 1024
    wordfiles = glob.glob(os.path.join(test_folder, '*.npy'))
    n_files = len(wordfiles)

    x_list = []
    y_list = []
    f_name_list = []
    for fidx in range(n_files):
        file = wordfiles[fidx]
        
        words = np.load(file)
        num_words = len(words)

        eos_arr = words[-1][None, ...]
        if num_words >= MAX_LEN - 2:
            print('too long!', num_words)
            continue

        x = words[:-1].copy()  #without EOS
        y = words[1:].copy()
        seq_len = len(x)
        print(' > seq_len:', seq_len)

        # pad with eos
        pad = np.tile(
            eos_arr, 
            (MAX_LEN-seq_len, 1))
        
        x = np.concatenate([x, pad], axis=0)
        y = np.concatenate([y, pad], axis=0)

        # collect
        if x.shape != (1024, 8):
            print(x.shape)
            exit()
        x_list.append(x.reshape(1, 1024,8))
        y_list.append(y.reshape(1, 1024,8))
        f_name_list.append(file)
    
    # x_final = np.array(x_list)
    # y_final = np.array(y_list)
    # f_name_list = np.array(f_name_list)
    return x_list, y_list, f_name_list



def take_embd_scratch(embd_net, test_folder):
    
    # unpack
    batch_size = 1
    train_x, train_y, f_name_list = compile_data(test_folder)
    num_batch = len(train_x) // batch_size

    

    # load model
    path_ckpt = 'exp/0309-1857' # path to ckpt dir
    loss = 30 # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
    print('[*] load model from:',  path_saved_ckpt)
    embd_net.load_state_dict(torch.load(path_saved_ckpt))

    embd_filename = 'temp/embd_'


    while train_x:

        batch_x = train_x.pop()
        batch_y = train_y.pop()
        fname = f_name_list.pop()
        
        batch_x = torch.from_numpy(batch_x).long().cuda()
        batch_y = torch.from_numpy(batch_y).long().cuda()
        
        
        h, y_type, layer_outputs  = embd_net.forward_hidden(batch_x)

        _, _, _, _, _, _, _, eight_y_ = embd_net.forward_output(layer_outputs[7], batch_y)

        np.save(embd_filename + fname.split('/')[-1], eight_y_.detach().cpu().numpy())
    
    return embd_filename

'''
