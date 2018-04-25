# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:57:50 2018

@author: ailingwang, yuanmingshi

Reference: https://github.com/udacity/AIND-VUI-Capstone

We offer two different kinds of prediction methods. One is with language model,
and the other is without a language model. To install language model, you need kenlm
and a dump of .lm file.

If we specify language model to be null, then it automatically generates all the predictions
in the testing/validating set you specify. You can also specicy the length of the audio you wanna
generate.

If we specify a language model, then you have to specify three things:
    - The path of a language model.
    - The range of texts you need to generate. Since using language models takes a long time (on average 1.2s per sentence), it's better not to do 10000 at one time.
    - The partition you need to test on (here, can be 'validation' or 'train', potentially, 'test' also, depending on your json content.
"""
import data_generator
from models import *
import numpy as np
from train_utils import *
from language_model import *
import argparse

def main():
    model_2 = cnn_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11,
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200)


    parser.add_argument('-m','--model', dest='model', help='model to run; default model is basic rnn', default='rnn')
    parser.add_argument('-mfcc', '--mfcc', dest='mfcc', help='using mfcc features', default=False)
    parser.add_argument('-p', '--pickle_path', dest='pickle_path', help='pickled path', default='model.pickle')
    parser.add_argument('-s', '--save_model_path', dest='save_model_path', help='save_model_path', default='model.h5')
    parser.add_argument('-u', '--units', dest='units', help='units', default=200)
    parser.add_argument('-l', '--layers', dest='layers', help='recurrent layers; only used for deep rnn model', default=3)
    parser.add_argument('-r', '--range', dest='range', help='number of transcriptions to generate', default=None)
    parser.add_argument('-lm', '--languageModel', dest='lm', help='language model path; on my local machine, \
                        use /home/whusym/DeepSpeech/models/lm/common_crawl_00.prune01111.trie.klm', default=None)
    parser.add_argument('-part', '--partition', dest='partition', help='partition to choose from; must be "train" or "validation"', default='validation')
    args = parser.parse_args()

    if args.mfcc == False:
        i_dim = 161
        use_spectrogram = True
    else:
        i_dim = 13

    # model 1: basic RNN
    if args.model == 'rnn':
        model = rnn_model(input_dim=i_dim,
                        units=args.units,
                        activation='relu')
    # model 2: bi-directional RNN
    elif args.model == 'brnn':
    	model = bidirectional_rnn_model(input_dim=i_dim,
    					units=args.units)
    # model 3: cnn rnn
    elif args.model == 'cnn_rnn':
    	model = cnn_rnn_model(input_dim=i_dim,
    						filters=200,
    						kernel_size=11,
    						conv_stride=2,
    						conv_border_mode='valid',
    						units=args.units)
    # model 4: time-distributed nn with lstm
    elif args.model == 'tdnn':
    	model =  TDNN_LSTM(input_dim=i_dim,
    					 output_dim=29)
    # model 5: deep rnn
    elif args.model == 'deep_rnn':
    	model = deep_rnn_model(input_dim=i_dim,
    					units=args.units,
    					recur_layers=args.layers)
    # model 6: deep speech 2 (by Baidu)
    elif args.model == 'ds2':
    	model = ds2_gru_model(input_dim=i_dim)
    else:
    	print ("Failed to specify a working model! Please choose among 'rnn', 'brnn', 'cnn_rnn', 'tdnn', 'deep_rnn', and 'ds2'.")
    	print ("For detailed information on these models, please check models.py file.")

    # Not using language model
    if not args.lm:
        if not args.range:
            predict_test(input_to_softmax=model, model_path='results/model.h5')
        else:
            predict_test(input_to_softmax=model, model_path='results/model.h5', audio_range=args.range)
    else:
        # specify a vocab map
        vocab = ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        lm = init_ext_scorer(args.lm, vocab)
        # make prediction
        all_pred = []
        all_trans = []
        all_audio_path = []
        if not args.range:
            print ("To use a language model, you have to specify a number that's less than the total \
                    number of files in your selection partition!")
        for i in range(args.range):
            loaded_matrix, transcr, audio_path = get_predictions(index=i,
                                    partition=parser.part,
                                    input_to_softmax=model_4,
                                    model_path='results/model_lstm_long_0421.h5',
                                    spectrogram_features=True)
            res = ctc_beam_search_decoder(loaded_matrix, vocab, 50, ext_scoring_func=lm)
            all_pred.append(res[0][1])
            all_trans.append(transcr)
            all_audio_path.append(audio_path)

if __name__ == "__main__":
    main()
