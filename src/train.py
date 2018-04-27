from data_generator import vis_train_features
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D,Conv2D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,Reshape)

from sample_models import *
from train_utils import train_model

if __name__ == "__main__":
    model_2 = cnn_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200)

    train_model(input_to_softmax=model_2, 
            pickle_path='model_2.pickle', 
            save_model_path='model_2.h5', 
            spectrogram=True) # change to False if you would like to use MFCC features
