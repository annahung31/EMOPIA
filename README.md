# EMOPIA

This is the official repository of **EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation**. The paper has been accepted by International Society for Music Information Retrieval Conference 2021.

- [Demo Page](https://annahung31.github.io/EMOPIA/)
- [Dataset at Zenodo (Coming soon)]()



## License
The EMOPIA dataset is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). It is provided primarily for research purposes and is prohibited to be used for commercial purposes. When sharing your result based on EMOPIA, any act that defames the original music owner is strictly prohibited.



# Emotion Classification

For the classification models and codes, please refer to [this repo](https://github.com/Dohppak/MIDI_Emotion_Classification).


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
1. Download the checkpoints and put them into `exp/`:
    - [Baseline](https://drive.google.com/file/d/1Q9vQYnNJ0hXBFwcxdWQgDNmzoW3MLl3h/view?usp=sharing)
    - [no-pretrained transformer](https://drive.google.com/file/d/1ZULJgBRu2Wb3jxFmGfAHP1v_tjoryFM7/view?usp=sharing)
    - [pretrained transformer](https://drive.google.com/file/d/19Seq18b2JNzOamEQMG1uarKjj27HJkHu/view?usp=sharing)

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
1. Download the data files from [HERE]().
    

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



## Authors

The paper is a co-work project with [Joann](https://github.com/joann8512), [SeungHeon](https://github.com/Dohppak) and Nabin. This repository is mentained by [Joann](https://github.com/joann8512) and me.



## Cite the dataset

```
@inproceedings{EMOPIA,
         author = {Hung, Hsiao-Tzu and Ching, Joann and Doh, Seungheon and Kim, Nabin and Nam, Juhan and Yang, Yi-Hsuan},
         title = {{MOPIA}: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation},
         booktitle = {Proc. Int. Society for Music Information Retrieval Conf.},
         year = {2021}
}
```
