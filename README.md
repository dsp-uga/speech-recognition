# speech-recognition
### Members:
* Yuanming Shi
* Ailing Wang


## Technology Used:
* Python

* Keras

* AIND-VUI-Capstone

* ASR-evaluation

* Baidu deepspeech

## Problem Statement

## Dataset

### LibriSpeech ASR corpus

We used LibriSpeech ASR corpus as our major training dataset.

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. 


To download the dataset and convert all .flac files to .wav files:

```
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
mv flac_to_wav.sh LibriSpeech
cd LibriSpeech
./flac_to_wav.sh
```
### DASS Corpus

## Technology Installation

* Keras
Create (and activate) a new environment with Python 3.6
```
conda create --name <environment_name> python=3.6 
source activate <environment_name>
```

Install Tensorflow
```
pip install tensorflow==1.1.0
```

Install Keras
```
sudo pip install keras
```


* AIND-VUI-Capstone Package

For more detailed local environment setup, please refer to https://github.com/udacity/AIND-VUI-Capstone/blob/master/README.md
```
git clone https://github.com/udacity/AIND-VUI-Capstone.git
cd AIND-VUI-Capstone
```
* ASR-evaluation

Installation:
```
pip install asr-evaluation
```

Commandline usage
```
wer <true_text.txt> <predicted_test.txt>
```

For more detailed information, please refer to https://github.com/belambert/asr-evaluation

* Baidu deepspeech

## Execution Steps


## Approaches We Tried

### Models We Explored

#### CNN-RNN Model
The first model we tried is CNN-RNN model, which include 1 Conv1D layer, 1 simple RNN layer and 1 Time distrubuted dense layer.

To use the model, simply pass cnn_rnn as argument to train.py file.

```
```

#### LSTM Model
We created a LSTM model which has 3 LSTM layer and 8 Time distributed dense layer. For the full dataset, which contains about 1000 hours of audio file, we increase the complexity of the model to 4 LSTM layer and 12 Time distributed layer. The structure of the model is inspired by https://arxiv.org/abs/1801.00059.

To use the model, pass tdnn or tdnn_large as argument to train.py file.


### Google VM

We created virtual machine instance in google cloud with 16 CPUs. It takes approximately 2 hours to train 1 epoch of the LSTM model on full dataset.


## Accuracy

## Reference

https://github.com/udacity/AIND-VUI-Capstone
http://www.openslr.org/12/

## Lisence
