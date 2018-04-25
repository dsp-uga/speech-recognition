"""
Defines a functions for training a NN.
from
"""

from data_generator import AudioGenerator
import _pickle as pickle
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda)
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from data_generator import AudioGenerator
from utils import int_sequence_to_text


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
        outputs=loss_out)
    return model

def train_model(input_to_softmax,
                pickle_path,
                save_model_path,
                train_json='.././json_dict/train_corpus.json',
                valid_json='.././json_dict/valid_corpus.json',
                minibatch_size=20,
                spectrogram=True,
                mfcc_dim=13,
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                epochs=20,
                verbose=1,
                sort_by_duration=False,
                max_duration=10.0):

    # create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(minibatch_size=minibatch_size,
        spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration,
        sort_by_duration=sort_by_duration)
    # add the training data to the generator
    audio_gen.load_train_data(train_json)
    audio_gen.load_validation_data(valid_json)
    # calculate steps_per_epoch
    num_train_examples=len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size
    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths)
    validation_steps = num_valid_samples//minibatch_size

    # add CTC loss to the NN specified in input_to_softmax
    model = add_ctc_loss(input_to_softmax)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # add checkpointer
    checkpointer = ModelCheckpoint(filepath='results/'+save_model_path, verbose=0)

    # train the model
    hist = model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=steps_per_epoch,
        epochs=epochs, validation_data=audio_gen.next_valid(), validation_steps=validation_steps,
        callbacks=[checkpointer], verbose=verbose)

    # save model loss
    with open('results/'+pickle_path, 'wb') as f:
        pickle.dump(hist.history, f)



def get_predictions(index, partition, input_to_softmax, model_path, spectrogram_features=True):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights

    return the predicted probability matrix (in a 2D matrix) and the ground truth
    """
    # load the train and test data
    data_gen = AudioGenerator(spectrogram=spectrogram_features)
    data_gen.load_train_data()
    data_gen.load_validation_data()

    # obtain the true transcription and the audio features
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    return (prediction[0], transcr, audio_path)

def predict_test(input_to_softmax, model_path, audio_range=len(audio_path)):
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_test_data()

    transcr = data_gen.test_texts
    audio_path = data_gen.test_audio_paths
    input_to_softmax.load_weights(model_path)
    predictions = []
    for i in range(len(audio_path)):  #default len(audio_path)):
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
