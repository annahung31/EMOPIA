# based on workspace/transformer/generate.ipynb

import subprocess
from pathlib import Path
import tempfile
import os
import pickle
import sys
import torch
import numpy as np
from midiSynth.synth import MidiSynth
import cog

sys.path.insert(0, "workspace/transformer")
from utils import write_midi
from models import TransformerModel


EMOTIONS = {
    "High valence, high arousal": 1,
    "Low valence, high arousal": 2,
    "Low valence, low arousal": 3,
    "High valence, low arousal": 4,
}


class Predictor(cog.Predictor):
    def setup(self):
        print("Loading dictionary...")
        path_dictionary = "dataset/co-representation/dictionary.pkl"
        with open(path_dictionary, "rb") as f:
            self.dictionary = pickle.load(f)
        event2word, self.word2event = self.dictionary

        n_class = []  # num of classes for each token
        for key in event2word.keys():
            n_class.append(len(event2word[key]))
        n_token = len(n_class)

        print("Loading model...")
        path_saved_ckpt = "exp/pretrained_transformer/loss_25_params.pt"
        self.net = TransformerModel(n_class, is_training=False)
        self.net.cuda()
        self.net.eval()

        self.net.load_state_dict(torch.load(path_saved_ckpt))

        self.midi_synth = MidiSynth()

    @cog.input(
        "emotion",
        type=str,
        default="High valence, high arousal",
        options=EMOTIONS.keys(),
        help="Emotion to generate for",
    )
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(self, emotion, seed):
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Prediction seed: {seed}")

        out_dir = Path(tempfile.mkdtemp())
        midi_path = out_dir / "out.midi"
        wav_path = out_dir / "out.wav"
        mp3_path = out_dir / "out.mp3"

        emotion_tag = EMOTIONS[emotion]
        res, _ = self.net.inference_from_scratch(
            self.dictionary, emotion_tag, n_token=8
        )
        try:
            write_midi(res, str(midi_path), self.word2event)
            self.midi_synth.midi2audio(str(midi_path), str(wav_path))
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(wav_path),
                    "-af",
                    "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                    str(mp3_path),
                ],
            )
            return mp3_path
        finally:
            midi_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)
