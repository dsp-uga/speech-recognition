'''
Some models are from

https://github.com/Udacity/AIND-VUI-capstone/blob/master/sample_models.py
, https://github.com/ShupingR/AIND-VUI-capstone/blob/master/sample_models.py,
and https://github.com/robmsmt/KerasDeepSpeech/blob/master/model.py
'''

from keras import backend as K
from keras.models import Model
from keras.layers import (ZeroPadding1D, BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

from keras.activations import relu


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)


    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    bn_rnn = input_data
    if recur_layers>=1:
        for i in range(recur_layers):
            # Add layer
            num = i + 1
            layer_name_rnn = "deep_rnn"+str(num)
            deep_rnn = GRU(units, activation='relu',
                       return_sequences=True, implementation=2,
                       name=layer_name_rnn)(bn_rnn)
            # Add batch normalization
            layer_name_bn_rnn = "bn_rnn"+str(num)
            bn_rnn = BatchNormalization(name=layer_name_bn_rnn)(deep_rnn)
    else:
        return(print("Error! The number of RNN layers must be >=1"))

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True,
                                  implementation=2, name='rnn'),
                              merge_mode='concat')(input_data)
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def ds2_gru_model(input_dim=161, fc_size=1024, rnn_size=512, output_dim=29, initialization='glorot_uniform',
                  conv_layers=1, gru_layers=1, use_conv=True):
    """ DeepSpeech 2 implementation
    Adopted from:
    https://github.com/robmsmt/KerasDeepSpeech/blob/master/model.py

    Architecture:
        Input Spectrogram TIMEx161
        1 Batch Normalisation layer on input
        1-3 Convolutional Layers
        1 Batch Normalisation layer
        1-7 BiDirectional GRU Layers
        1 Batch Normalisation layer
        1 Fully connected Dense
        1 Softmax output
    Details:
       - Uses Spectrogram as input rather than MFCC
       - Did not use BN on the first input
       - Network does not dynamically adapt to maximum audio size in the first convolutional layer. Max conv
          length padded at 2048 chars, otherwise use_conv=False
    Reference:
        https://arxiv.org/abs/1512.02595
    """

    # K.set_learning_phase(1)
    def clipped_relu(x):
        return relu(x, max_value=20)
    input_data = Input(shape=(None, input_dim), name='the_input')
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(input_data)

    if use_conv:
        conv = ZeroPadding1D(padding=(0, 2048))(x)
        for l in range(conv_layers):
            x = Conv1D(filters=fc_size, name='conv_{}'.format(l+1), kernel_size=11, padding='valid', activation='relu', strides=2)(conv)
    else:
        for l in range(conv_layers):
            x = TimeDistributed(Dense(fc_size, name='fc_{}'.format(l + 1), activation='relu'))(x)  # >>(?, time, fc_size)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

    for l in range(gru_layers):
        x = Bidirectional(GRU(rnn_size, name='fc_{}'.format(l + 1), return_sequences=True, activation='relu', kernel_initializer=initialization),
                      merge_mode='sum')(x)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

    # Last Layer 5+6 Time Dist Dense Layer & Softmax
    x = TimeDistributed(Dense(fc_size, activation=clipped_relu))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(x)

    # labels = K.placeholder(name='the_labels', ndim=1, dtype='int32')
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length=lambda x:x
    print(model.summary())
    return model

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

    # Add batch normalization
    bn_td2 = BatchNormalization(name='bn_td2')(time_dense5)
    lstm2 = LSTM(50, activation='tanh',return_sequences=True)(bn_td2)

    time_dense6 = TimeDistributed(Dense(50))(lstm2)
    time_dense7 = TimeDistributed(Dense(50))(time_dense6)

    # Add batch normalization
    bn_td3 = BatchNormalization(name='bn_td3')(time_dense7)
    lstm3 = LSTM(50, activation='tanh',return_sequences=True)(bn_td3)

    time_dense = TimeDistributed(Dense(output_dim))(lstm3)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
