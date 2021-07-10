# EMOPIA

EMOPIA  (pronounced  ‘yee-mò-pi-uh’)  dataset is a  shared multi-modal (audio and MIDI) database focusing on perceived emotion in pop piano music, to facilitate research8on  various  tasks  related  to  music  emotion. The dataset contains 1,087 music clips from 387 songs and clip-level emotion  labels  annotated by four dedicated  annotators. Since the clips are not restricted to one clip per song, they can also be used for song-level analysis.   

The detail of the methodology  for  building  the  dataset please refer to our paper.  

# Cite us

```
@inproceedings{EMOPIA,
         author = {Hsiao-Tzu Hung, Joann Ching, Seungheon Doh, Nabin Kim, Juhan Nam, Yi-Hsuan Yang},
         title = {EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation},
         booktitle = {Proc. Int. Society for Music Information Retrieval Conf.},
         year = {2021}
}
```

# Emotion Classification

For the classification models and codes, please refer to [this repo](https://github.com/Dohppak/MIDI_Emotion_Classification).


# Conditional Generation



<h2>Q1 (High Valence, high Arousal)</h2>

<table class="audio-table">
  <tbody>
    <tr>
      <td>Baseline</td>
      <td><audio controls=""><source src="./assets/audio_samples/1_lstm+GA/Q1/gen_Q1_1.mp3" type="audio/mpeg" /></audio></td>
      <td><audio controls=""><source src="./assets/audio_samples/1_lstm+GA/Q1/gen_Q1_2.mp3" type="audio/mpeg" /></audio></td>
      <td><audio controls=""><source src="./assets/audio_samples/1_lstm+GA/Q1/gen_Q1_3.mp3" type="audio/mpeg" /></audio></td>
    </tr>
    <tr>
      <td>Transformer w/o pre-training</td>
      <td><audio controls=""><source src="./assets/audio_samples/2_Transformer/Q1/Q1_1.mp3" type="audio/mpeg" /></audio></td>
      <td><audio controls=""><source src="./assets/audio_samples/2_Transformer/Q1/Q1_2.mp3" type="audio/mpeg" /></audio></td>
      <td><audio controls=""><source src="./assets/audio_samples/2_Transformer/Q1/Q1_3.mp3" type="audio/mpeg" /></audio></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td>Transformer w/ pre-training</td>
      <td><audio controls=""><source src="./assets/audio_samples/3_Pre-trained_Transformer/Q1/Q1_1.mp3" type="audio/mpeg" /></audio></td>
      <td><audio controls=""><source src="./assets/audio_samples/3_Pre-trained_Transformer/Q1/Q1_2.mp3" type="audio/mpeg" /></audio></td>
      <td><audio controls=""><source src="./assets/audio_samples/3_Pre-trained_Transformer/Q1/Q1_3.mp3" type="audio/mpeg" /></audio></td>
    </tr>
  </tfoot>
</table>