'''
This code is from
https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/representations/uncond/cp/compile.py
'''


import os
import json
import pickle
import numpy as np
import ipdb


TEST_AMOUNT = 50
WINDOW_SIZE = 512
GROUP_SIZE = 2    #7
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = 'linear' # 'linear', 'XL'
print('[config] MAX_LEN:', MAX_LEN)

broken_list = [
'Q1_8izVTDgBQPc_0.mid.pkl.npy',  'Q1_uqRLEByE6pU_1.mid.pkl.npy',  'Q3_nBIls0laAAU_0.mid.pkl.npy',
'Q1_8rupdevqfuI_0.mid.pkl.npy',  'Q2_9v2WSpn4FCw_1.mid.pkl.npy',  'Q3_REq37pDAm3A_3.mid.pkl.npy',
'Q1_aYe-2Glruu4_3.mid.pkl.npy',  'Q2_BzqX-9TA-GY_2.mid.pkl.npy',  'Q3_RL_cmmNVLfs_0.mid.pkl.npy',
'Q1_FfwKrQyQ7WU_2.mid.pkl.npy',  'Q2_ivCNV47tsRw_1.mid.pkl.npy',  'Q3_wfXSdMsd4q8_4.mid.pkl.npy',
'Q1_GY3f6ckBVkA_1.mid.pkl.npy',  'Q2_RiQMuhk_SuQ_1.mid.pkl.npy',  'Q3_wqc8iqbDsGM_0.mid.pkl.npy',
'Q1_Jn9r0avp0fY_1.mid.pkl.npy',  'Q3_bbU31JLtlug_1.mid.pkl.npy',  'Q4_OUb9uaOlWAM_0.mid.pkl.npy',
'Q1_NGE9ynTJABg_0.mid.pkl.npy',  'Q3_c6CwY8Gbw0c_2.mid.pkl.npy',  'Q4_V3Y9L4UOcpk_1.mid.pkl.npy',
'Q1_QwsQ8ejbMKg_1.mid.pkl.npy',  'Q3_kDGmND1BgmA_1.mid.pkl.npy']



def traverse_dir(
        root_dir,
        extension=('mid', 'MID'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file in broken_list:
                continue
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list
    

if __name__ == '__main__':
    # paths
    path_root = './'
    path_indir = os.path.join( path_root, 'words')

    # load all words
    wordfiles = traverse_dir(
            path_indir,
            extension=('npy'))
    n_files = len(wordfiles)

    # init
    x_list = []
    y_list = []
    mask_list = []
    seq_len_list = []
    num_groups_list = []
    name_list = []

    # process
    for fidx in range(n_files):
        print('--[{}/{}]-----'.format(fidx+1, n_files))
        file = wordfiles[fidx]
        words = np.load(file)
        
        num_words = len(words)

        eos_arr = words[-1][None, ...]

        if num_words >= MAX_LEN - 2: # 2 for room
            words = words[:MAX_LEN-2]

        # arrange IO
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
        mask = np.concatenate(
            [np.ones(seq_len), np.zeros(MAX_LEN-seq_len)])

        # collect
        if x.shape != (1024, 8):
            print(x.shape)
            exit()
        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)
        seq_len_list.append(seq_len)
        num_groups_list.append(int(np.ceil(seq_len/WINDOW_SIZE)))
        name_list.append(file)

    # sort by length (descending) 
    zipped = zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list)
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip( 
                                    *sorted(zipped, key=lambda x: -x[0])) 

    print('\n\n[Finished]')
    print(' compile target:', COMPILE_TARGET)
    if COMPILE_TARGET == 'XL':
        # reshape
        x_final = np.array(x_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        y_final = np.array(y_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
    elif COMPILE_TARGET == 'linear':
        
        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError('Unknown target:', COMPILE_TARGET)

    # check
    num_samples = len(seq_len_list)
    print(' >   count:', )
    print(' > x_final:', x_final.shape)
    print(' > y_final:', y_final.shape)
    print(' > mask_final:', mask_final.shape)
    
    train_idx = []

    # validation filename map
    fn2idx_map = {
        'fn2idx': dict(),
        'idx2fn': dict(),
    }

    # training filename map
    train_fn2idx_map = {
        'fn2idx': dict(),
        'idx2fn': dict(),
    }

    name_list = [x.split('/')[-1].split('.')[0] for x in name_list]
    # run split
    train_cnt = 0
    for nidx, n in enumerate(name_list):
        
        train_idx.append(nidx)    
        train_fn2idx_map['fn2idx'][n] = train_cnt
        train_fn2idx_map['idx2fn'][train_cnt] = n
        train_cnt += 1

    train_idx = np.array(train_idx)
    
    # save train map
    path_train_fn2idx_map = os.path.join(path_root,  'train_fn2idx_map.json')
    with open(path_train_fn2idx_map, 'w') as f:
        json.dump(train_fn2idx_map, f)

    # save train
    path_train = os.path.join(path_root, 'train_data_{}'.format(COMPILE_TARGET))
    path_train += '.npz'
    print('save to', path_train)
    np.savez(
        path_train, 
        x=x_final[train_idx],
        y=y_final[train_idx],
        mask=mask_final[train_idx],
        seq_len=np.array(seq_len_list)[train_idx],        
        num_groups=np.array(num_groups_list)[train_idx]
    )
   
    print('---')
    print(' > train x:', x_final[train_idx].shape)

