import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Directory where plots will be saved
PLOTS_DIR = './results_base'

def plot_logits(xs, ys, top_neurons):
    for n in top_neurons:
        plot_logit_and_save(xs, ys, n)

def plot_logit_and_save(xs, ys, neuron_index):
    sentiment_unit = xs[:,neuron_index]

    # plt.title('Distribution of Logit Values')
    plt.ylabel('Number of Phrases')
    plt.xlabel('Value of the Sentiment Neuron')
    plt.hist(sentiment_unit[ys == -1], bins=50, alpha=0.5, label='Negative Phrases')
    plt.hist(sentiment_unit[ys ==  1], bins=50, alpha=0.5, label='Positive Phrases')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "neuron_" + str(neuron_index) + '.png'))
    plt.clf()

def plot_weight_contribs(coef):
    plt.title('Values of Resulting L1 Penalized Weights')
    plt.tick_params(axis='both', which='major')

    # Normalize weight contributions
    norm = np.linalg.norm(coef)
    coef = coef/norm

    plt.plot(range(len(coef[0])), coef.T)
    plt.xlabel('Neuron (Feature) Index')
    plt.ylabel('Neuron (Feature) weight')
    plt.savefig(os.path.join(PLOTS_DIR, "weight_contribs.png"))
    plt.clf()
