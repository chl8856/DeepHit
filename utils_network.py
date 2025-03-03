"""
First implemented: 01/25/2018
  > For survival analysis on longitudinal dataset
By CHANGHEE LEE

Modifcation List:
        - 08/07/2018: weight regularization for FC_NET is added
"""

import tensorflow as tf


# CONSTRUCT MULTICELL FOR MULTI-LAYER RNNS
def create_rnn_cell(num_units, num_layers, keep_prob, RNN_type):
    """
    GOAL         : create multi-cell (including a single cell) to construct multi-layer RNN
    num_units    : number of units in each layer
    num_layers   : number of layers in MulticellRNN
    keep_prob    : keep probabilty [0, 1]  (if None, dropout is not employed)
    RNN_type     : either 'LSTM' or 'GRU'
    """
    cells = []
    for _ in range(num_layers):
        if RNN_type == "GRU":
            cell = tf.keras.layers.GRUCell(num_units)
        elif RNN_type == "LSTM":
            cell = tf.keras.layers.LSTMCell(num_units)
        if keep_prob is not None:
            cell = tf.keras.layers.Dropout(1 - keep_prob)(cell)
        cells.append(cell)
        cell = tf.keras.layers.RNN(cells, return_sequences=True, return_state=True)

    return cell


# EXTRACT STATE OUTPUT OF MULTICELL-RNNS
def create_concat_state(state, num_layers, RNN_type):
    """
    GOAL	     : concatenate the tuple-type tensor (state) into a single tensor
    state        : input state is a tuple ofo MulticellRNN (i.e. output of MulticellRNN)
                   consist of only hidden states h for GRU and hidden states c and h for LSTM
    num_layers   : number of layers in MulticellRNN
    RNN_type     : either 'LSTM' or 'GRU'
    """
    for i in range(num_layers):
        if RNN_type == "LSTM":
            tmp = state[i][1]  # i-th layer, h state for LSTM
        elif RNN_type == "GRU":
            tmp = state[i]  # i-th layer, h state for GRU
        else:
            print("ERROR: WRONG RNN CELL TYPE")

        if i == 0:
            rnn_state_out = tmp
        else:
            rnn_state_out = tf.concat([rnn_state_out, tmp], axis=1)

    return rnn_state_out


# FEEDFORWARD NETWORK
def create_FCNet(
    inputs, num_layers, h_dim, h_fn, o_dim, o_fn, w_init, keep_prob=1.0, w_reg=None
):
    """
    GOAL             : Create FC network with different specifications
    inputs (tensor)  : input tensor
    num_layers       : number of layers in FCNet
    h_dim  (int)     : number of hidden units
    h_fn             : activation function for hidden layers (default: tf.nn.relu)
    o_dim  (int)     : number of output units
    o_fn             : activation function for output layers (defalut: None)
    w_init           : initialization for weight matrix (defalut: Xavier)
    keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
    """
    # default active functions (hidden: relu, out: None)
    if h_fn is None:
        h_fn = tf.nn.relu
    if o_fn is None:
        o_fn = None

    # default initialization functions (weight: Xavier, bias: None)
    if w_init is None:
        w_init = tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )  # Xavier initialization

    for layer in range(num_layers):
        if num_layers == 1:
            out = tf.keras.layers.Dense(
                o_dim,
                activation=o_fn,
                kernel_initializer=w_init,
                kernel_regularizer=w_reg,
            )(inputs)
        else:
            if layer == 0:
                h = tf.keras.layers.Dense(
                    h_dim,
                    activation=h_fn,
                    kernel_initializer=w_init,
                    kernel_regularizer=w_reg,
                )(inputs)
                if keep_prob is not None:
                    h = tf.keras.layers.Dropout(1 - keep_prob)(h)

            elif layer > 0 and layer != (num_layers - 1):  # Hidden layers
                h = tf.keras.layers.Dense(
                    h_dim,
                    activation=h_fn,
                    kernel_initializer=w_init,
                    kernel_regularizer=w_reg,
                )(h)
                if keep_prob is not None:
                    h = tf.keras.layers.Dropout(1 - keep_prob)(h)

            else:  # Last layer
                out = tf.keras.layers.Dense(
                    o_dim,
                    activation=o_fn,
                    kernel_initializer=w_init,
                    kernel_regularizer=w_reg,
                )(h)

    return out
