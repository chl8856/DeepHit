#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 19:16:06 2021

@author: igor
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from import_data import f_get_fc_mask2, f_get_fc_mask3
from class_DeepHit import Model_DeepHit


##############################################################################
data_mode = 'CUSTOMERS' 


# this saves the current hyperparameters
def save_logging(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            f.write('%s:%s\n' % (key, value))

# this open can calls the saved hyperparameters
def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here    
    return data


from import_data import import_dataset_SYNTHETIC, import_dataset_METABRIC, import_dataset_CUSTOMERS 

if data_mode == 'METABRIC':
    DIM, DATA, MASK = import_dataset_METABRIC()
elif data_mode == 'SYNTHETIC':    
    DIM, DATA, MASK = import_dataset_SYNTHETIC()
elif data_mode == 'CUSTOMERS':    
    DIM, DATA, MASK = import_dataset_CUSTOMERS()

seed = 1234


##############################################################################

if data_mode == 'CUSTOMERS':
    out_itr = 0

# Load the saved optimised hyperparameters
in_path = data_mode + '/results/'
in_hypfile = in_path + '/itr_' + str(out_itr) + '/hyperparameters_log.txt'
in_parser = load_logging(in_hypfile)

# Forward the hyperparameters
mb_size                     = in_parser['mb_size']

iteration                   = in_parser['iteration']

keep_prob                   = in_parser['keep_prob']
lr_train                    = in_parser['lr_train']

h_dim_shared                = in_parser['h_dim_shared']
h_dim_CS                    = in_parser['h_dim_CS']
num_layers_shared           = in_parser['num_layers_shared']
num_layers_CS               = in_parser['num_layers_CS']

if in_parser['active_fn'] == 'relu':
    active_fn                = tf.nn.relu
elif in_parser['active_fn'] == 'elu':
    active_fn                = tf.nn.elu
elif in_parser['active_fn'] == 'tanh':
    active_fn                = tf.nn.tanh
else:
    print('Error!')


initial_W                   = tf.contrib.layers.xavier_initializer()

# Weights for the loss functions that also can be tuned
# Default values: 1.0
alpha                       = in_parser['alpha']  #for log-likelihood loss
beta                        = in_parser['beta']  #for ranking loss

# # Factor of 1.2 was used in the paper in order to have enough time-horizon
num_Category       = int(440+1) #int(np.max(DATA[1]) * 1.2) 
  
# # Number of events (censoring is not included)
num_Event          = int(len(np.unique(DATA[2])) - 1) 

# # Input dimension
x_dim              = np.shape(DATA[0])[1]

data, time, event = DATA
# # Based on the data, mask1 and mask2 needed to calculate the loss functions
# # To calculate loss 1 - log-likelihood loss
mask1              = f_get_fc_mask2(time, event, num_Event, num_Category)
# # To calculate loss 2 - cause-specific ranking loss
mask2              = f_get_fc_mask3(time, -1, num_Category)

# Create the dictionaries 
# For the input settings
input_dims                  = { 'x_dim'         : x_dim,
                                'num_Event'     : num_Event,
                                'num_Category'  : num_Category}

# For the hyperparameters
network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                'h_dim_CS'          : h_dim_CS,
                                'num_layers_shared'    : num_layers_shared,
                                'num_layers_CS'    : num_layers_CS,
                                'active_fn'      : active_fn,
                                'initial_W'         : initial_W }

# Create the DeepHit network architecture
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# Restoring the trained model
saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

# Final prediction on the test set 
churn_probabilities = model.predict(data)

# Calculating loyalty curve
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
colors = ['green', 'blue', 'yellow', 'orange', 'red']
max_days = 365
total_persons = len(churn_probabilities)
Y = np.zeros((len(percentiles), max_days)) + total_persons
X = np.array([i for i in range(max_days)])
for x in X:
    churn_probabilities_at_x = churn_probabilities[np.where(time == x)]
    for i_p in range(len(percentiles)):
        Y[i_p, x:] -= np.count_nonzero(churn_probabilities_at_x > percentiles[i_p])
Y[np.where(Y < 0)] = 0

d = {'Day': X}
for i, y in enumerate(Y):
    d[f"{int(100*percentiles[i])} %"] = y

df = pd.DataFrame(d)
df.to_csv(in_path + '/itr_' + str(out_itr) + '/loyalty.csv')

plt.title("Loyalty Curve")

for i in range(len(percentiles)):
    plt.plot(X, Y[i], color=colors[i], label=f"{int(100*percentiles[i])} %")

plt.fill_between(X, Y[0], color=colors[0])
for i in range(1, len(colors)):
    plt.fill_between(X, Y[i-1], Y[i], color=colors[i])

plt.legend()
plt.show()