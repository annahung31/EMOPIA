
# Dataset processing

## If you want to use the processed data used in our paper...
Download the npz files from [here](https://drive.google.com/file/d/17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP/view?usp=sharing), or download via gdown:  

```
gdown --id 17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP
unzip co-representation.zip
```


## If you want to prepare data from scratch...
The data pre-processing used in our paper is basically the same as [Compound-word-transformer](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md). So please refer to this repository to prepare data from scratch.

## If you want to build your own dictionary and build CP words....

If you want to make use of EMOPIA along with other dataset, you might need to prepare your own dictionary. In that way, you need `REMI_tokens` of each midi file.  

1. Download [EMOPIA 2.1](https://zenodo.org/record/5151045#.YQaNfVMzZoQ). Inside the folders, `CP_events` are the CP events processed using [Compound-word-transformer](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md) process. 
2. Put `CP_events` in your `events` folder.
3. Put other dataset's event files also in the `events` folder.
3. Along with other dataset's event files, run the following scripts:

a. transfer the events to CP words, and build the dictionary.
```
python event2words.py
```
b. Compile the words file to npz file for training.
```
python compile.py
```
* Please note that I don't split data into train/test set, I use all the data for training bacause the task is to generate music from scratch and no need for validation.
* The broken list in the compile.py is samples that encountered some issues during preprocessing and I just skip them.