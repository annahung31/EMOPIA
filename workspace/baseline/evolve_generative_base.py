import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json
import pickle
import argparse
import numpy as np
import tensorflow as tf

from midi_generator import generate_midi
from train_classifier import encode_sentence
from train_classifier import get_activated_neurons
from train_generative import build_generative_model

GEN_MIN = -1
GEN_MAX =  1

# Directory where trained model will be saved
TRAIN_DIR = "./trained"

def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.uniform(0, 1) < mutation_rate:
            individual[i] = np.random.uniform(GEN_MIN, GEN_MAX)

def crossover(parent_a, parent_b, ind_size):
    # Averaging crossover
    return (parent_a + parent_b)/2

def reproduce(mating_pool, new_population_size, ind_size, mutation_rate):
    new_population = np.zeros((new_population_size, ind_size))

    for i in range(new_population_size):
        a = np.random.randint(len(mating_pool));
        b = np.random.randint(len(mating_pool));

        new_population[i] = crossover(mating_pool[a], mating_pool[b], ind_size);

    # Mutate new children
    np.apply_along_axis(mutation, 1, new_population, mutation_rate)

    return new_population;

def roulette_wheel(population, fitness_pop):
    # Normalize fitnesses
    norm_fitness_pop = fitness_pop/np.sum(fitness_pop)

    # Here all the fitnesses sum up to 1
    r = np.random.uniform(0, 1)

    fitness_so_far = 0
    for i in range(len(population)):
        fitness_so_far += norm_fitness_pop[i]

        if r < fitness_so_far:
            return population[i]

    return population[-1]

def select(population, fitness_pop, mating_pool_size, ind_size, elite_rate):
    mating_pool = np.zeros((mating_pool_size, ind_size))

    # Apply roulete wheel to select mating_pool_size individuals
    for i in range(mating_pool_size):
        mating_pool[i] = roulette_wheel(population, fitness_pop)

    # Apply elitism
    assert elite_rate >= 0 and elite_rate <= 1
    elite_size = int(np.ceil(elite_rate * len(population)))
    elite_idxs = np.argsort(-fitness_pop)

    for i in range(elite_size):
        r = np.random.randint(0, mating_pool_size)
        mating_pool[r] = elite_idxs[i]

    return mating_pool

def calc_fitness(individual, gen_model, cls_model, char2idx, idx2char, layer_idx, sentiment, runs=30):
    encoding_size = gen_model.layers[layer_idx].units
    generated_midis = np.zeros((runs, encoding_size))

    # Get activated neurons
    sentneuron_ixs = get_activated_neurons(cls_model)
    assert len(individual) == len(sentneuron_ixs)

    # Use individual gens to override model neurons
    override = {}
    for i, ix in enumerate(sentneuron_ixs):
        override[ix] = individual[i]

    # Generate pieces and encode them using the cell state of the generative model
    for i in range(runs):
        midi_text = generate_midi(gen_model, char2idx, idx2char, seq_len=64, layer_idx=layer_idx, override=override)
        generated_midis[i] = encode_sentence(gen_model, midi_text, char2idx, layer_idx)

    midis_sentiment = cls_model.predict(generated_midis).clip(min=0)
    return 1.0 - np.sum(np.abs(midis_sentiment - sentiment))/runs

def evaluate(population, gen_model, cls_model, char2idx, idx2char, layer_idx, sentiment):
    fitness = np.zeros((len(population), 1))

    for i in range(len(population)):
        fitness[i] = calc_fitness(population[i], gen_model, cls_model, char2idx, idx2char, layer_idx, sentiment)

    return fitness

def evolve(pop_size, ind_size, mut_rate, elite_rate, epochs):
    # Create initial population
    population = np.random.uniform(GEN_MIN, GEN_MAX, (pop_size, ind_size))

    # Evaluate initial population
    fitness_pop = evaluate(population, gen_model, cls_model, char2idx, idx2char, opt.cellix, sent)
    print("--> Fitness: \n", fitness_pop)

    for i in range(epochs):
        print("-> Epoch", i)

        # Select individuals via roulette wheel to form a mating pool
        mating_pool = select(population, fitness_pop, pop_size, ind_size, elite_rate)

        # Reproduce matin pool with crossover and mutation to form new population
        population = reproduce(mating_pool, pop_size, ind_size, mut_rate)

        # Calculate fitness of each individual of the population
        fitness_pop = evaluate(population, gen_model, cls_model, char2idx, idx2char, opt.cellix, sent)
        print("--> Fitness: \n", fitness_pop)

    return population, fitness_pop

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='evolve_generative.py')
    parser.add_argument('--genmodel', type=str, default='./trained', help="Generative model to evolve.")
    parser.add_argument('--clsmodel', type=str, default='./trained/classifier_ckpt.p', help="Classifier model to calculate fitness.")
    parser.add_argument('--ch2ix', type=str, default='./trained/char2idx.json', help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, default=256, help="Embedding size.")
    parser.add_argument('--units', type=int, default=512, help="LSTM units.")
    parser.add_argument('--layers', type=int, default=4, help="LSTM layers.")
    parser.add_argument('--cellix', type=int, default=4, help="LSTM layer to use as encoder.")
    #parser.add_argument('--sent', type=int, default=2, help="Desired sentiment.")
    parser.add_argument('--popsize', type=int, default=10, help="Population size.")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs to run.")
    parser.add_argument('--mrate', type=float, default=0.1, help="Mutation rate.")
    parser.add_argument('--elitism', type=float, default=0.1, help="Elitism in percentage.")

    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char,idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild generative model from checkpoint
    gen_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    gen_model.load_weights(tf.train.latest_checkpoint(opt.genmodel))
    gen_model.build(tf.TensorShape([1, None]))

    # Load classifier model
    with open(opt.clsmodel, "rb") as f:
        cls_model = pickle.load(f)

    # Set individual size to the number of activated neurons
    sentneuron_ixs = get_activated_neurons(cls_model)
    ind_size = len(sentneuron_ixs)

    # Evolve for Positive(1)/Negative(1)
    sents = [1, 2, 3, 4]
    for sent in sents: 
        print('evolving for {}'.format(sent))
        population, fitness_pop = evolve(opt.popsize, ind_size, opt.mrate, opt.elitism, opt.epochs)

        # Get best individual
        best_idx = np.argmax(fitness_pop)
        best_individual = population[best_idx]

        # Use best individual gens to create a dictionary with cell values
        neurons = {}
        for i, ix in enumerate(sentneuron_ixs):
            neurons[str(ix)] = best_individual[i]


        # Persist dictionary with cell values
        if sent == 1:
            sent = 'Q1'
        elif sent == 2:
            sent = 'Q2'
        elif sent == 3:
            sent = 'Q3'
        else:
            sent = 'Q4'

        with open(os.path.join(TRAIN_DIR, "neurons_" + sent + ".json"), "w") as f:
            json.dump(neurons, f)