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
* 

## Execution Steps


## Approaches We Tried

## Accuracy

## Reference

https://github.com/udacity/AIND-VUI-Capstone
http://www.openslr.org/12/

## Lisence
