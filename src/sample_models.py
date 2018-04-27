## From https://github.com/ShupingR/AIND-VUI-capstone/blob/master/sample_models.py

from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
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
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
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
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
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
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, 
                                  implementation=2, name='rnn'),
                              merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers=1, output_dim=29):
    """ Build a deep network for speech 
    """  
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    bn_rnn = bn_cnn 
    
    for i in range(recur_layers):
        num = i + 1
        # Add bidirecitonal layer
        layer_name = "rnn"+str(i+1)
        bidir_rnn = Bidirectional(GRU(units, return_sequences=True, 
                                      implementation=2, dropout=0.2,
                                      name=layer_name),
                                  merge_mode='concat')(bn_rnn)
        # Add batch normalization
        batch_name = "bn"+layer_name
        bn_rnn = BatchNormalization(name = batch_name)(bidir_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
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
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: x
    print(model.summary())
    return model

def stack_LSTM(input_dim, output_dim=29):
    """ Build a deep network for speech 
    """  
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    

    lstm1a = LSTM(250,activation="tanh",return_sequences=True)(input_data)
    lstm1b = LSTM(250,activation="tanh",return_sequences=True)(lstm1a)
    time_dense1 = TimeDistributed(Dense(200))(lstm1b)

    # Add batch normalization
    bn_td1 = BatchNormalization(name='bn_td1')(time_dense1)
    lstm2a = LSTM(150, activation='tanh',return_sequences=True)(bn_td1)
    lstm2b = LSTM(150, activation='tanh',return_sequences=True)(lstm2a)
    
    time_dense2 = TimeDistributed(Dense(120))(lstm2b)

    # Add batch normalization
    bn_td2 = BatchNormalization(name='bn_td2')(time_dense2)    
    lstm3a = LSTM(80, activation='tanh',return_sequences=True)(bn_td2)
    
    time_dense = TimeDistributed(Dense(output_dim))(lstm3a)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: x
    print(model.summary())
    return model
