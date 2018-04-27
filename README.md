# Automatic Speech Recognition on the Digital Archive of the Southern Speech

This project is impolemented over the course of three weeks as the final project of the CSCI 8360 Data Science Practicum class offered in Spring 2018 at the University of Georgia. For course webpage, [click here](http://dsp-uga.github.io/sp18/schedule.html). 

## Prerequisites:

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/)
- [Tensorflow](http://www.tensorflow.org)
- [Keras](http://keras.io)
- [Jupyter Notebook](http://jupyter.org/) (for presentation purposes)
- [ASR Evaluation](https://github.com/belambert/asr-evaluation) (for evaluting WER)
- Swig Decoder (for decoding the outputs of the network by using a language model. For instructions, check the wiki page.)

For other required libraries, please check `environment.yml` file.

### Google VM

We created virtual machine instance in google cloud with 16 CPUs. It takes approximately 2 hours to train 1 epoch of the LSTM model on full dataset.

## Problem Statement

## Datasets

### LibriSpeech ASR corpus

We use LibriSpeech ASR corpus as our major training dataset. LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey.It is publicly available. To access all the different datasets files, [click here.](http://openslr.org/12) 

You also need to convert all .flac files of LibriSpeech to .wav files. There are a lot of available scripts online for doing so. [Here is an example to do so](https://github.com/udacity/AIND-VUI-Capstone/blob/master/flac_to_wav.sh). 

### DASS Corpus

## How to install and use

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





## Accuracy

## Team Members:
* Yuanming Shi
* Ailing Wang
See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for detailed contributions.

## Reference

https://github.com/udacity/AIND-VUI-Capstone
http://www.openslr.org/12/

## License
