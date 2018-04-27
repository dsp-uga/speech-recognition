import data_generator 
from sample_models import *
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text

from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def TDNN_LSTM(input_dim, output_dim=29):
    """ Build a deep network for speech 
    """  
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    time_dense1 = TimeDistributed(Dense(150))(input_data)
    time_dense2 = TimeDistributed(Dense(140))(time_dense1)
    time_dense3 = TimeDistributed(Dense(130))(time_dense2)

    # Add batch normalization
    bn_td1 = BatchNormalization(name='bn_td1')(time_dense3)
    lstm1 = LSTM(100, activation='tanh',return_sequences=True)(bn_td1)
    
    time_dense4 = TimeDistributed(Dense(80))(lstm1)
    time_dense5 = TimeDistributed(Dense(70))(time_dense4)

    lstm3 = LSTM(50, activation='tanh',return_sequences=True)(time_dense5)

    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(lstm3)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: x
    print(model.summary())
    return model

model = TDNN_LSTM(161, output_dim=29)

from train_utils import train_model

train_model(input_to_softmax=model, 
            pickle_path='model_tdnn.pickle', 
            save_model_path='model_tdnn.h5', 
            spectrogram=True,
            epochs = 100)
