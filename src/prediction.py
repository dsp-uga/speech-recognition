# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:57:50 2018

@author: ailingwang

Reference: https://github.com/udacity/AIND-VUI-Capstone

"""
import data_generator 
from sample_models import *
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text


def predict_test(input_to_softmax, model_path):
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()

    transcr = data_gen.valid_texts
    audio_path = data_gen.valid_audio_paths
    input_to_softmax.load_weights(model_path)
    predictions = []
    for i in range(10):#len(audio_path)):
        data_point = data_gen.normalize(data_gen.featurize(audio_path[i]))
        
        prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
        output_length = [input_to_softmax.output_length(data_point.shape[0])] 
        pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
        pred = ''.join(int_sequence_to_text(pred_ints))
        predictions.append(pred)

    predictions = ''.join(predictions)
    transcr = transcr[:10]
    transcr = ''.join(transcr)
    with open("predictions/predictions.txt", "w") as output:
        output.write(str(predictions))
    with open("predictions/truescr.txt", "w") as output:
        output.write(str(transcr))

if __name__ == "__main__":
    model_2 = cnn_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200)
                        
    predict_test(input_to_softmax=model_2, model_path='results/model_lstm_long_0424.h5')
