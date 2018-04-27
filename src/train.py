'''
Main file for training the selected model. The basic command is:
python train.py <-m model_name> <-mfcc False> <-p pickled path> <-s save_model_path> <-e epochs> <-u units> <-l recurrent layers>
'''
from models import *
from train_utils import train_model
import argparse


def info():
    '''
    Print system info.
    '''
    print ('Python version:')
    print (sys.version)
    print ('Tensorflow version:')
    print (tf.__version__)
    print ('Tensorflow GPU Support')
    print (tf.test.gpu_device_name())


def main():
    parser = argparse.ArgumentParser(description='Speech Recognition in Librispeech',
            argument_default=argparse.SUPPRESS)

    options = parser.add_subparsers()
    # train.py info
    op = options.add_parser('info', description='print system info')
    op.set_defaults(func=info)


    parser.add_argument('-m','--model', dest='model', help='model to run; default model is basic rnn', default='rnn')
    parser.add_argument('-mfcc', '--mfcc', dest='mfcc', help='using mfcc features', default=False)
    parser.add_argument('-p', '--pickle_path', dest='pickle_path', help='pickled path', default='model.pickle')
    parser.add_argument('-s', '--save_model_path', dest='save_model_path', help='save_model_path', default='model.h5')
    parser.add_argument('-e', '--epochs', dest='epochs', help='epochs', default=20)
    parser.add_argument('-u', '--units', dest='units', help='units', default=200)
    parser.add_argument('-l', '--layers', dest='layers', help='recurrent layers; only used for deep rnn model', default=3)
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
        
    # model 7: stacked lstm
    elif args.model == 'stack_lstm':
        model = stack_LSTM(input_dim=i_dim,
    					 output_dim=29)
    else:
    	print ("Failed to specify a working model! Please choose among 'rnn', 'brnn', 'cnn_rnn', 'tdnn', 'deep_rnn', and 'ds2'.")
    	print ("For detailed information on these models, please check models.py file.")
        
    

    train_model(input_to_softmax=model,
                pickle_path=args.pickle_path,
                save_model_path=args.save_model_path,
                spectrogram=use_spectrogram,
                epochs=args.epochs)

    print ('Training done! Check ./results folder for saved models.')


if __name__ == '__main__':
    main()
