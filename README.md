# Automatic Speech Recognition on the Digital Archive of the Southern Speech
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This project is impolemented over the course of three weeks as the final project of the CSCI 8360 Data Science Practicum class offered in Spring 2018 at the University of Georgia. For course webpage, [click here](http://dsp-uga.github.io/sp18/schedule.html). This project has benefited from the depositories from [Udacity's NanoDegree](https://github.com/udacity/AIND-VUI-Capstone), [@robmsmt](https://github.com/robmsmt/KerasDeepSpeech), [Baidu's Bay Area DL School](https://github.com/baidu-research/ba-dls-deepspeech), and [Baidu's PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech). Huge thanks to these people who make these sources publicly available! 

## Prerequisites:

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/)
- [Tensorflow](http://www.tensorflow.org)
- [Keras](http://keras.io)
- [Jupyter Notebook](http://jupyter.org/) (for presentation purposes)
- [ASR Evaluation](https://github.com/belambert/asr-evaluation) (for evaluting WER)
- [Swig](http://www.swig.org/) (used for decoding the outputs of the network by using a language model. To check more detailed instructions, check the wiki page.)

For other required libraries, please check `environment.yml` file.

### Google VM Hardware Specs

We created virtual machine instance in google cloud with 16v CPU and 64GB of RAM. It takes approximately 2 hours to train 1 epoch of the LSTM model on full dataset (train-960 of LibriSpeech). With 8v CPU, 50GB of RAM, and a Nvidia Tesla P100 GPU, it takes around 750 seconds to train 1 epoch of Baidu's DeepSpeech2 Model on the train-100 set of LibriSpeech. 


## Deliverables 
Our presentation comes in the form of a Jupyter Notebook. The notebook file is under the [src](./src) folder and it is named `demo.ipynb`. To see its html version, please navigate to the [presentation](./presentation) folder and check `demo.html` and its dependencies.


## Datasets

### LibriSpeech ASR corpus
We use LibriSpeech ASR corpus as our major training dataset. LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey.It is publicly available. To access all the different datasets files, [click here.](http://openslr.org/12) 

You also need to convert all .flac files of LibriSpeech to .wav files. There are a lot of available scripts online for doing so. [Here is an example on how to do so](https://github.com/udacity/AIND-VUI-Capstone/blob/master/flac_to_wav.sh). 

### DASS (Digital Archive of Southern Speech) Corpus
DASS is an audio corpus that records 64 interviews (3-4 hours each) of Southern speeches featuring dialects in eight Southern states with a mixture of ethnicities, ages, social classes, and education levels. DASS provides fruitful resources for researchers to work on. It would be interesting to see how well the model trained on general North American English corpora performs on these Southern speeches. The audio data is also publicly accessible from http://www.lap.uga.edu/, and it is also available via the Linguistic Data Consortium (https://catalog.ldc.upenn.edu/LDC2016S05). 


## Environment Settings

0. Install dependencies

There are several sound-processing tools you need to install, with `libav` being the main one (because `soundfile` library in Python needs it to run):
	  - __Linux__: `sudo apt-get install libav-tools`
	  - __Mac__: `brew install libav`

Also, as we have mentioned before `swig` is also used as an option in integrating a language model to generate recognized texts. Detailed instructions on installing it could be found in our [Wiki](https://github.com/dsp-uga/speech-recognition/wiki) (to navigate to the wiki, you can also press `g` `w` on your keyboard) .

1. Clone this repository.
```
$ git clone https://github.com/dsp-uga/speech-recognition.git
$ cd speech-recognition
```

2. Create a conda environment based on `environments.yml` offered in this repository. 
```
$ conda env create -f environments.yml -n <environment_name> python=3.6
$ source activate <environment_name>
```
It will build a conda environment with the name you specify. 

3. Download all the audio files into the [data](./data) folder and organize them as it is instructed there. 

4. Generate json dictionaries 

Json dictionaries record the paths, lengths, and transciptions of all the audio files. In order to train your model, two json files are necessary -- `train_corpus.json` and `valid_corpus.json`. If you also have a test set to test upon, you also need another json file -- `test_corpus.json`. (And of course, you can change their names and specify them in your training/testing process). In our system, the json dictionaries are stored in the [json_dict](./json_dict) folder. You can check the README file there to see what a json dict should look like in detail. To generate training and validation these json dictionaries:
```
cd src
python create_desc_json.py .././data/LibriSpeech/dev-clean/ .././json_dict/train_corpus.json
python create_desc_json.py .././data/LibriSpeech/test-clean/ .././json_dict/valid_corpus.json
```
Note that these two commands are for the `dev-clean` and `test-clean` datasets in LibriSpeech. It is assumed that you already have these files downloaded in the `data` folder. 

To generate testing json dictionary of DASS:
```
cd src
python create_test_desc_json.py .././data/DASS .././json_dict/test_corpus.json
```

5. Train a model
We offer these following models for users to train on. Here is the command to run it:
```
cd src
python train.py <-m model_name> <-mfcc False> <-p pickled_path> <-s save_model_path> <-e epochs> <-u units> <-l recurrent layers>
```

#### CNN-RNN Model
The first model we tried is CNN-RNN model, which include 1 Conv1D layer, 1 simple RNN layer and 1 Time distrubuted dense layer.

#### Bi-directional RNN (BRNN) Model
BRNN is an improvement from Vanilla RNN. We can also stack recurrent layers together and make it Deep BRNN. Its usage in speech recognition is pioneered by Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Check out their famous paper from: https://arxiv.org/abs/1303.5778.

#### LSTM Model
We created a LSTM model which has 3 LSTM layer and 8 Time distributed dense layer. For the full dataset, which contains about 1000 hours of audio file, we increase the complexity of the model to 4 LSTM layer and 12 Time distributed layer. The structure of the model is inspired by https://arxiv.org/abs/1801.00059.

#### Baidu's DeepSpeech 2 Model 
The model consists of 1-3 1D or 2D convolutional layers, 7 RNN layers. The original paper can be found at https://arxiv.org/abs/1512.02595, and we use the Keras implementation from [here](https://github.com/robmsmt/KerasDeepSpeech/blob/master/model.py).

6. Use a trained model for prediction
To make predictions, users can use the following command: 
```
cd src
python predict.py <-m model_name> <-mfcc False> <-p pickled_path> <-s save_model_path> <-u units> <-l recurrent layers> <-r range> <-lm languageModel> <-part partitionOfDatasets>
```

Although the command seems cumbersome, we have provided a lot of default settings so users do not need to specify most of the arguments in practice. To see how the detailed help on each of the argument, check out the source [here](https://github.com/dsp-uga/speech-recognition/blob/Documentation/src/predict.py).

## How to evaluate

We use Word Error Rate (WER) to evaluate our system. We use the the open-sourced `asr-evaluation` package to do so.

Installation (if `asr-evalatuion` is not installed in your customized conda environment already):
```
pip install asr-evaluation
```

Commandline usage (in the examples here, we put our results under the `prediction` folder):
```
cd prediction
wer <true_text.txt> <predicted_test.txt>
```

For more detailed information, please refer to https://github.com/belambert/asr-evaluation


## Results

So far, the best WER on LibriSpeech is around 80% and the best WER on DASS is around 102%. We are still in the process of improving our models. 

## TODO
- More improvements for the architectures of these models.
- More hyperparameter tuning for the models
- Better language models.

## How to Contribute
We welcome any kind of contribution. If you want to contribute, just create a ticket!

## Team Members:
* Yuanming Shi, Institute for Artificial Intelligence, The University of Georgia
* Ailing Wang, Department of Computer Science, The University of Georgia 

See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for detailed contributions by each team member.

## License
MIT

## Reference

https://github.com/udacity/AIND-VUI-Capstone

https://github.com/baidu-research/ba-dls-deepspeech

https://github.com/PaddlePaddle/DeepSpeech

https://github.com/robmsmt/KerasDeepSpeech

http://www.openslr.org/12/

http://lap3.libs.uga.edu/u/jstanley/vowelcharts/

https://catalog.ldc.upenn.edu/LDC2012S03

http://www.lap.uga.edu/
