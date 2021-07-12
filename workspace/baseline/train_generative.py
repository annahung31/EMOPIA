import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"\
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import json
import argparse
import numpy      as np
import tensorflow as tf
import numpy as np


import midi_encoder as me

# Directory where the checkpoints will be saved
TRAIN_DIR = "./trained/"

#print('Version: ', tf.__version__)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#print("Check: ", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


def generative_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def build_generative_model(vocab_size, embed_dim, lstm_units, lstm_layers, batch_size, dropout=0):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))

    for i in range(max(1, lstm_layers)):
        model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=True, stateful=True, dropout=dropout, recurrent_dropout=dropout))

    model.add(tf.keras.layers.Dense(vocab_size))

    return model

def build_char2idx(train_vocab, test_vocab):
    # Merge train and test vocabulary
    vocab = list(train_vocab | test_vocab)
    vocab.sort()

    # Calculate vocab size
    vocab_size = len(vocab)

    # Create dict to support char to index conversion
    char2idx = { char:i for i,char in enumerate(vocab) }

    # Save char2idx encoding as a json file for generate midi later
    with open(os.path.join(TRAIN_DIR, "char2idx.json"), "w") as f:
        json.dump(char2idx, f)

    return char2idx, vocab_size

def build_dataset(text, char2idx, seq_length, batch_size, buffer_size=10000):
    text_as_int = np.array([char2idx[c] for c in text.split(" ")])
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(__split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset

def train_generative_model(model, train_dataset, test_dataset, epochs, learning_rate):
    # Compile model with given optimizer and defined loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=generative_loss)

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(TRAIN_DIR, "generative_ckpt_{epoch}")
    my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)]
    
    
    return model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=my_callbacks)

def __split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_generative.py')
    parser.add_argument('--train', type=str, default='../data/train/', help="Train dataset.")
    parser.add_argument('--test' , type=str, default='../data/test/', help="Test dataset.")
    parser.add_argument('--model', type=str, required=False, help="Checkpoint dir.")
    parser.add_argument('--embed', type=int, default=256, help="Embedding size.")
    parser.add_argument('--units', type=int, default=512, help="LSTM units.")
    parser.add_argument('--layers', type=int, default=4, help="LSTM layers.")
    parser.add_argument('--batch', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=30, help="Epochs.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence lenght.")
    parser.add_argument('--lrate', type=float, default=0.00001, help="Learning rate.")
    parser.add_argument('--drop', type=float, default=0.05, help="Dropout.")
    opt = parser.parse_args()

    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    
    # Encode midi files as text with vocab
    train_text, train_vocab = me.load(opt.train)
    test_text, test_vocab = me.load(opt.test)

    # Build dictionary to map from char to integers
    char2idx, vocab_size = build_char2idx(train_vocab, test_vocab)

    # Build dataset from encoded unlabelled midis
    train_dataset = build_dataset(train_text, char2idx, opt.seqlen, opt.batch)
    test_dataset = build_dataset(test_text, char2idx, opt.seqlen, opt.batch)

    # Build generative model
    generative_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, opt.batch, opt.drop)

    if opt.model:
        # If pre-trained model was given as argument, load weights from disk
        print("Loading weights from {}...".format(opt.model))
        generative_model.load_weights(tf.train.latest_checkpoint(opt.model))

    # Train model
    history = train_generative_model(generative_model, train_dataset, test_dataset, opt.epochs, opt.lrate)
    print("Total of {} epochs used for training.".format(len(history.history['loss'])))
    loss_hist = history.history['loss']
    print("Best loss from history: ", np.min(loss_hist))
