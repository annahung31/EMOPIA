
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sampling

# import fast_transformers_local
from fast_transformers.builders import TransformerEncoderBuilder as TransformerEncoderBuilder_local
from fast_transformers.builders import RecurrentEncoderBuilder as RecurrentEncoderBuilder_local 
from fast_transformers.masking import TriangularCausalMask as TriangularCausalMask_local


D_MODEL = 512
N_LAYER = 12  
N_HEAD = 8   

################################################################################
# Model
################################################################################

def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params



class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class TransformerModel(nn.Module):
    def __init__(self, n_token, is_training=True, data_parallel=False):
        super(TransformerModel, self).__init__()
        self.data_parallel = data_parallel
        # --- params config --- #
        self.n_token = n_token     # == n_class
        self.d_model = D_MODEL 
        self.n_layer = N_LAYER #
        self.dropout = 0.1
        self.n_head = N_HEAD #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        if len(self.n_token) == 8:
            self.emb_sizes = [128, 256, 64, 32, 512, 128, 128, 128]
        elif len(self.n_token) == 9:
            self.emb_sizes = [128, 256, 64, 32, 512, 128, 128, 128, 128]   #128

        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type      = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch     = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration  = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity  = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.word_emb_emotion   = Embeddings(self.n_token[7], self.emb_sizes[7])
        if len(self.n_token) == 9:
            self.word_emb_key       = Embeddings(self.n_token[8], self.emb_sizes[8])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        
        # linear 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

         # encoder
        if is_training:
            # encoder (training)
            self.get_encoder('encoder')


        else:
            # encoder (inference)
            print(' [o] using RNN backend.')
            self.get_encoder('autoregred')

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        # individual output
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type     = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])
        self.proj_emotion = nn.Linear(self.d_model, self.n_token[7])
        if len(self.n_token) == 9:
            self.proj_key = nn.Linear( self.d_model, self.n_token[8])
        

    def compute_loss(self, predict, target, loss_mask):
        if self.data_parallel:
            loss = self.loss_func(predict, target)
            loss = loss * loss_mask
            return torch.sum(loss), torch.sum(loss_mask)
        else:
            loss = self.loss_func(predict, target)
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
            return loss



    def forward(self, x, target, loss_mask):


        h, y_type = self.forward_hidden(x, is_training=True)
        

        if len(self.n_token) == 9:
            y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, y_key, emo_embd = self.forward_output(h, target)
        else:
            y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, emo_embd = self.forward_output(h, target)
        


        # reshape (b, s, f) -> (b, f, s)
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_type      = y_type[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)
        y_emotion   = y_emotion[:, ...].permute(0, 2, 1)
        if len(self.n_token) == 9:
            y_key       = y_key[:, ...].permute(0, 2, 1)
        
        # loss
        loss_tempo = self.compute_loss(
                y_tempo, target[..., 0], loss_mask)
        loss_chord = self.compute_loss(
                y_chord, target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(
                y_barbeat, target[..., 2], loss_mask)
        loss_type = self.compute_loss(
                y_type,  target[..., 3], loss_mask)
        loss_pitch = self.compute_loss(
                y_pitch, target[..., 4], loss_mask)
        loss_duration = self.compute_loss(
                y_duration, target[..., 5], loss_mask)
        loss_velocity = self.compute_loss(
                y_velocity, target[..., 6], loss_mask)
        loss_emotion = self.compute_loss(
                y_emotion,  target[..., 7], loss_mask)
        
        
        if len(self.n_token) == 9:
            loss_key = self.compute_loss(
                    y_key,  target[..., 8], loss_mask)
        
            return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity, loss_emotion, loss_key

        else:
            return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity, loss_emotion



    def get_encoder(self, TYPE):
        if TYPE == 'encoder':
            self.transformer_encoder = TransformerEncoderBuilder_local.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()
            
        
        elif TYPE == 'autoregred':
            self.transformer_encoder = RecurrentEncoderBuilder_local.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()



    def forward_hidden(self, x, memory=None, is_training=False):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''
        
        # embeddings
        emb_tempo =    self.word_emb_tempo(x[..., 0])
        emb_chord =    self.word_emb_chord(x[..., 1])
        emb_barbeat =  self.word_emb_barbeat(x[..., 2])
        emb_type =     self.word_emb_type(x[..., 3])
        emb_pitch =    self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])

        emb_emotion = self.word_emb_emotion(x[..., 7])

        if len(self.n_token) == 9:
            emb_key = self.word_emb_key(x[..., 8])
        
        # same emotion class have same emb_emotion
        
            embs = torch.cat(
                [
                    emb_tempo,
                    emb_chord,
                    emb_barbeat,
                    emb_type,
                    emb_pitch,
                    emb_duration,
                    emb_velocity,
                    emb_emotion,
                    emb_key
                ], dim=-1)

        else:
            embs = torch.cat(
                [
                    emb_tempo,
                    emb_chord,
                    emb_barbeat,
                    emb_type,
                    emb_pitch,
                    emb_duration,
                    emb_velocity,
                    emb_emotion
                
                ], dim=-1)


        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)
        
        
        # assert False
        layer_outputs = []
        # transformer
        if is_training:
            # mask
            attn_mask = TriangularCausalMask_local(pos_emb.size(1), device=x.device)

            h = self.transformer_encoder(pos_emb, attn_mask) # y: b x s x d_model
            

            # project type
            y_type = self.proj_type(h)


            return h, y_type

        else:
            pos_emb = pos_emb.squeeze(0)
            
            # self.get_encoder('autoregred')
            # self.transformer_encoder.cuda()
            h, memory = self.transformer_encoder(pos_emb, memory=memory) # y: s x d_model
            
            # project type
            y_type = self.proj_type(h)
            
            return h, y_type, memory


    def forward_output(self, h, y):
        '''
        for training
        '''
        # tf_skip_emption = self.word_emb_emotion(y[..., 7])
        tf_skip_type = self.word_emb_type(y[..., 3])

        emo_embd = h[:, 0]
        
        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)
        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_emotion = self.proj_emotion(y_)

        if len(self.n_token) == 9:
            y_key = self.proj_key(y_)
        
            return  y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, y_key, emo_embd

        else:
            return  y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, emo_embd




    def froward_output_sampling(self, h, y_type, is_training=False):
        '''
        for inference
        '''
        
        # sample type
        y_type_logit = y_type[0, :]   # token class size
        cur_word_type = sampling(y_type_logit, p=0.90, is_training=is_training)  # int
        if cur_word_type is None:
            return None, None

        if is_training:
            type_word_t = cur_word_type.long().unsqueeze(0).unsqueeze(0)
        else:
            type_word_t = torch.from_numpy(
                    np.array([cur_word_type])).long().cuda().unsqueeze(0)        # shape = (1,1)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)                # shape = (1, embd_size)
        
        
        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        # project other
        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)

        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_emotion = self.proj_emotion(y_)
        
            
        
        # sampling gen_cond
        cur_word_tempo =    sampling(y_tempo, t=1.2, p=0.9, is_training=is_training)
        cur_word_barbeat =  sampling(y_barbeat, t=1.2, is_training=is_training)
        cur_word_chord =    sampling(y_chord, p=0.99, is_training=is_training)
        cur_word_pitch =    sampling(y_pitch, p=0.9, is_training=is_training)
        cur_word_duration = sampling(y_duration, t=2, p=0.9, is_training=is_training)
        cur_word_velocity = sampling(y_velocity, t=5, is_training=is_training)        
        
        if len(self.n_token) == 9:
            y_key = self.proj_key(y_)
            cur_word_key      = sampling(y_key, t=1.2, is_training=is_training)    

            curs = [
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity,
                cur_word_key
            ]

        else:
            curs = [
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity
            ]

        if None in curs:
            return None, None



        if is_training:
            cur_word_emotion = torch.from_numpy(np.array([0])).long().cuda().squeeze(0)
            # collect
            next_arr = torch.tensor([
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_type,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity,
                cur_word_emotion
                ])        
        
        else:
            cur_word_emotion = 0
            
            
            # collect
            if len(self.n_token) == 9:
                next_arr = np.array([
                    cur_word_tempo,
                    cur_word_chord,
                    cur_word_barbeat,
                    cur_word_type,
                    cur_word_pitch,
                    cur_word_duration,
                    cur_word_velocity,
                    cur_word_emotion,
                    cur_word_key
                    ])      
            else:
                next_arr = np.array([
                    cur_word_tempo,
                    cur_word_chord,
                    cur_word_barbeat,
                    cur_word_type,
                    cur_word_pitch,
                    cur_word_duration,
                    cur_word_velocity,
                    cur_word_emotion
                    ])        
            
        return next_arr, y_emotion





    def inference_from_scratch(self, dictionary, emotion_tag, key_tag=None, n_token=8, display=True):
        event2word, word2event = dictionary
        

        classes = word2event.keys()
        
        
        def print_word_cp(cp):
            
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')
        
        generated_key = None



        if n_token == 9:
            
            if key_tag:
                
                target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag, 0]
                target_key     = [0, 0, 0, 4, 0, 0, 0, 0, key_tag]
                
                init = np.array([
                    target_emotion,  # emotion
                    target_key,
                    [0, 0, 1, 2, 0, 0, 0, 0, 0] # bar
                ])
            
            else:
                target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag, 0]
                init = np.array([
                    target_emotion,  # emotion
                    [0, 0, 1, 2, 0, 0, 0, 0, 0] # bar
                ])

        elif n_token == 8:
            target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag]
            
            init = np.array([
                target_emotion,  # emotion
                [0, 0, 1, 2, 0, 0, 0, 0] # bar
            ])


        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            memory = None
            h = None
            
            cnt_bar = 1
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')

            if n_token == 9 and  key_tag is None:
                # Emotion token
                step = 0
                if display:
                    print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                h, y_type, memory = self.forward_hidden(
                            input_, memory, is_training=False)

                #generate KEY
                next_arr, y_emotion = self.froward_output_sampling(h, y_type)
                if next_arr is None:
                    return None, None

                generated_key = next_arr[-1]  
                final_res.append(next_arr[None, ...])
                if display:
                    print_word_cp(next_arr)
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_  = input_.unsqueeze(0).unsqueeze(0)
                h, y_type, memory = self.forward_hidden(
                            input_, memory, is_training=False)

                # init bar
                step = 1
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                h, y_type, memory = self.forward_hidden(
                            input_, memory, is_training=False)

                

            else:
                for step in range(init.shape[0]):

                    print_word_cp(init[step, :])
                    input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                    final_res.append(init[step, :][None, ...])
                    
                    h, y_type, memory = self.forward_hidden(
                            input_, memory, is_training=False)
                    
                    

            
            print('------ generate ------')
            while(True):
                # sample others
                next_arr, y_emotion = self.froward_output_sampling(h, y_type)
                if next_arr is None:
                    return None, None
                    
                final_res.append(next_arr[None, ...])
                
                if display:
                    print('bar:', cnt_bar, end= '  ==')
                    print_word_cp(next_arr)
                
                # forward
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_  = input_.unsqueeze(0).unsqueeze(0)
                h, y_type, memory = self.forward_hidden(
                    input_, memory, is_training=False)

                # end of sequence
                if word2event['type'][next_arr[3]] == 'EOS':
                    break
                
                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
           
        
        return final_res, generated_key

