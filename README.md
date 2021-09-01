
<div align="center">
    <img src=./docs/img/emopia.png width=200x>
</div>

This is the official repository of **EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation**. The paper has been accepted by International Society for Music Information Retrieval Conference 2021.

- [Paper on Arxiv](https://arxiv.org/abs/2108.01374)
- [Demo Page](https://annahung31.github.io/EMOPIA/)
- [Interactive demo and Docker image on Replicate](https://replicate.ai/annahung31/emopia)
- [Dataset at Zenodo](https://zenodo.org/record/5090631#.YPPo-JMzZz8)

* Note: We release the transcribed MIDI files. As for the audio part, due to the copyright issue, we will only release the YouTube ID of the tracks and the timestamp of them. You might use [open source crawler](https://github.com/ytdl-org/youtube-dl) to get the audio file.


## Use EMOPIA by MusPy
1. install muspy
```
pip install muspy
```
2. Use it in your script

```
import muspy

emopia = muspy.EMOPIADataset("data/emopia/", download_and_extract=True)
emopia.convert()
music = emopia[0]
print(music.annotations[0].annotation)
```
You can get the label of the piece of music:

```
{'emo_class': '1', 'YouTube_ID': '0vLPYiPN7qY', 'seg_id': '0'}
```
* `emo_class`: ['1', '2', '3', '4']
* `YouTube_ID`: the YouTube ID of this piece of music
* `seg_id`: means this piece of music is the `i`th piece we take from this song. (zero-based). 

For more usage please refer to [MusPy](https://github.com/salu133445/muspy).


# Emotion Classification

For the classification models and codes, please refer to [this repo](https://github.com/SeungHeonDoh/EMOPIA_cls).


# Conditional Generation

## Environment

1. Install PyTorch and fast transformer:
    - torch==1.7.0 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4).)
    - fast transformer :

        ```
        pip install --user pytorch-fast-transformers 
        ```
        or refer to the original [repository](https://github.com/idiap/fast-transformers)

2. Other requirements:

    pip install -r requirements.txt


## Usage

### Inference

**Option 1:**  

You can directly run the [generate script](./workspace/transformer/generate.ipynb) to generate pieces of musics and listen to them.


**Option 2:**  
Or you might follow the steps as below.

1. Download the checkpoints and put them into `exp/`
    * Manually:  
        - [Baseline](https://drive.google.com/file/d/1Q9vQYnNJ0hXBFwcxdWQgDNmzoW3MLl3h/view?usp=sharing)
        - [no-pretrained transformer](https://drive.google.com/file/d/1ZULJgBRu2Wb3jxFmGfAHP1v_tjoryFM7/view?usp=sharing)
        - [pretrained transformer](https://drive.google.com/file/d/19Seq18b2JNzOamEQMG1uarKjj27HJkHu/view?usp=sharing)

    * By commend: (install gdown: `pip install gdown`) 
        ```
        #baseline:
        gdown --id 1Q9vQYnNJ0hXBFwcxdWQgDNmzoW3MLl3h --output exp/baseline.zip

        # no-pretrained transformer
        gdown --id 1ZULJgBRu2Wb3jxFmGfAHP1v_tjoryFM7 --output exp/no-pretrained_transformer.zip

        # pretrained transformer
        gdown --id 19Seq18b2JNzOamEQMG1uarKjj27HJkHu --output exp/pretrained_transformer.zip
        ```



2. Inference options:

* `num_songs`: number of midis you want to generate.
* `out_dir`: the folder where the generated midi will be saved. If not specified, midi files will be saved to `exp/MODEL_YOU_USED/gen_midis/`.
* `task_type`: the task_type needs to be the same as the task specified during training.  
    - '4-cls' for 4 class conditioning
    - 'Arousal' for only conditioning on arousal
    - 'Valence' for only conditioning on Valence
    - 'ignore' for not conditioning

*  `emo_tag`: the target class of emotion you want to assign.
    - If the task_type is '4-cls', emo_tag can be: 1,2,3,4, which refers to Q1, Q2, Q3, Q4.
    - If the task_type is 'Arousal', emo_tag can be: `1`, `2`. `1` for High arousal, `2` for Low arousal.
    - If the task_type is 'Valence', emo_tag can be: `1`, `2`. `1` for High Valence, `2` for Low Valence.
    

3. Inference

    ```
    python main_cp.py --mode inference --task_type 4-cls --load_ckt CHECKPOINT_FOLDER --load_ckt_loss 25 --num_songs 10 --emo_tag 1 
    ```

### Train the model by yourself
1. Prepare the data follow the [steps](https://github.com/annahung31/EMOPIA/tree/main/dataset).
    

2. training options:  

* `exp_name`: the folder name that the checkpoints will be saved.
* `data_parallel`: use data_parallel to let the training process faster. (0: not use, 1: use)
* `task_type`: the conditioning task:
    - '4-cls' for 4 class conditioning
    - 'Arousal' for only conditioning on arousal
    - 'Valence' for only conditioning on Valence
    - 'ignore' for not conditioning

    a. Only train on EMOPIA: (`no-pretrained transformer` in the paper)

        python main_cp.py --path_train_data emopia --exp_name YOUR_EXP_NAME --load_ckt none
    
    b. Pre-train the transformer on `AILabs17k`:  
    
        python main_cp.py --path_train_data ailabs --exp_name YOUR_EXP_NAME --load_ckt none --task_type ignore
    
    c. fine-tune the transformer on `EMOPIA`:
        For example, you want to use the pre-trained model stored in `0309-1857` with loss= `30` to fine-tune:

        python main_cp.py --path_train_data emopia --exp_name YOUR_EXP_NAME --load_ckt 0309-1857 --load_ckt_loss 30

### Baseline
1. The baseline code is based on the work of [Learning to Generate Music with Sentiment](https://github.com/lucasnfe/music-sentneuron)

2. According to the author, the model works best when it is trained with 4096 neurons of LSTM, but takes 12 days for training. Therefore, due to the limit of computational resource, we used the size of 512 neurons instead of 4096.

3. In order to use this as evaluation against our model, the target emotion classes is expanded to 4Q instead of just positive/negative.

## Authors

The paper is a co-working project with [Joann](https://github.com/joann8512), [SeungHeon](https://github.com/SeungHeonDoh) and Nabin. This repository is mentained by [Joann](https://github.com/joann8512) and me.


## License
The EMOPIA dataset is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). It is provided primarily for research purposes and is prohibited to be used for commercial purposes. When sharing your result based on EMOPIA, any act that defames the original music owner is strictly prohibited.  


The hand drawn piano in the logo comes from [Adobe stock](https://stock.adobe.com/tw/images/one-line-piano-instrument-design-hand-drawn-minimalistic-style-vector-illustration/327942843). The author is [Burak](https://stock.adobe.com/tw/contributor/206697762/burak?load_type=author&prev_url=detail). I purchased it under [standard](https://stock.adobe.com/tw/license-terms) license.

## Cite the dataset

```
@inproceedings{{EMOPIA},
         author = {Hung, Hsiao-Tzu and Ching, Joann and Doh, Seungheon and Kim, Nabin and Nam, Juhan and Yang, Yi-Hsuan},
         title = {{MOPIA}: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation},
         booktitle = {Proc. Int. Society for Music Information Retrieval Conf.},
         year = {2021}
}
```
