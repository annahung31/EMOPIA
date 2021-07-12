import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model
from train_classifier import preprocess_sentence

GENERATED_DIR = './generated'

def override_neurons(model, layer_idx, override):
    h_state, c_state = model.get_layer(index=layer_idx).states

    c_state = c_state.numpy()
    for neuron, value in override.items():
        c_state[:,int(neuron)] = int(value)

    model.get_layer(index=layer_idx).states = (h_state, tf.Variable(c_state))

def sample_next(predictions, k):
    # Sample using a categorical distribution over the top k midi chars
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    if np.random.uniform(0, 1) < .5:
        predicted_id = top_k_choices[0]
    else:
        p_choices = tf.math.softmax(top_k_values[1:]).numpy()
        predicted_id = np.random.choice(top_k_choices[1:], 1, p=p_choices)[0]

    return predicted_id

def process_init_text(model, init_text, char2idx, layer_idx, override):
    model.reset_states()

    for c in init_text.split(" "):
        # Run a forward pass
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)

            # override sentiment neurons
            override_neurons(model, layer_idx, override)

            predictions = model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", s)

    return predictions

def generate_midi(model, char2idx, idx2char, init_text="", seq_len=256, k=3, layer_idx=-2, override={}):
    # Add front and end pad to the initial text
    init_text = preprocess_sentence(init_text)

    # Empty midi to store our results
    midi_generated = []

    # Process initial text
    predictions = process_init_text(model, init_text, char2idx, layer_idx, override)

    # Here batch size == 1
    model.reset_states()
    for i in range(seq_len):
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0).numpy()

        # Sample using a categorical distribution over the top k midi chars
        predicted_id = sample_next(predictions, k)

         # Append it to generated midi
        midi_generated.append(idx2char[predicted_id])

        # override sentiment neurons
        override_neurons(model, layer_idx, override)

        #Run a new forward pass
        input_eval = tf.expand_dims([predicted_id], 0)
        predictions = model(input_eval)

    return init_text + " " + " ".join(midi_generated)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, default='./trained', help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, default='./trained/char2idx.json', help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, default=256, help="Embedding size.")
    parser.add_argument('--units', type=int, default=512, help="LSTM units.")
    parser.add_argument('--layers', type=int, default=4, help="LSTM layers.")
    parser.add_argument('--seqinit', type=str, default="\n", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=512, help="Sequence lenght.")
    parser.add_argument('--cellix', type=int, default=4, help="LSTM layer to use as encoder.")
    parser.add_argument('--override', type=str, default="./trained/neurons_Q1.json", help="JSON file with neuron values to override.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Load override dict from json file
    override = {}

    try:
        with open(opt.override) as f:
            override = json.load(f)
    except FileNotFoundError:
        print("Override JSON file not provided.")

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char,idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild model from checkpoint
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(opt.model))
    model.build(tf.TensorShape([1, None]))

    if not os.path.exists(GENERATED_DIR):
        os.makedirs(GENERATED_DIR)
    # Generate 5 midis
    for i in range(100):
        # Generate a midi as text
        print("Generate midi {}".format(i))
        midi_txt = generate_midi(model, char2idx, idx2char, opt.seqinit, opt.seqlen, layer_idx=opt.cellix, override=override)
        

        me.write(midi_txt, os.path.join(GENERATED_DIR, "generated_Q1_{}.mid".format(i)))
