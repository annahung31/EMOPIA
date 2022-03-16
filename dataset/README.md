
# Dataset processing

## If you want to use the processed data used in our paper...
Download the npz files from [here](https://drive.google.com/file/d/17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP/view?usp=sharing), or download via gdown:  

```
gdown --id 17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP
unzip co-representation.zip
```


## If you want to prepare data from scratch...
The data pre-processing used in our paper is basically the same as [Compound-word-transformer](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md). The difference is the emotion token part.


1. Run step 1-3 of [Compound-word-transformer dataset processing](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md).
2. Change the path in the following scripts and run:

a. quantize everything, add EOS, and prepare the emotion label from the filename.

```
python midi2corpus.py
```

b. transfer the corpus file to CP events.

```
python corpus2events.py
```

c. transfer the events to CP words, and build the dictionary.
```
python event2words.py
```
d. Compile the words file to npz file for training.
```
python compile.py
```
* Please note that I don't split data into train/test set, I use all the data for training bacause the task is to generate music from scratch and no need for validation.
* The broken list in the compile.py is samples that encountered some issues during preprocessing and I just skip them.