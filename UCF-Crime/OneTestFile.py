import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.ops import math_ops
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.optimizers import SGD
from ESN import EchoStateRNNCell
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense, Activation, Dropout, Bidirectional,LSTM, GRU, Flatten, Concatenate
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import os
################################## MODEL #####################################

random_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print("seed: ", random_seed)

nb_classes = 2
batches = 16
stime = 16
epochs = 400
num_units = 16
num_inputs = 1024
num_outputs = 2

activation = lambda x: math_ops.tanh(x)
cell = EchoStateRNNCell(units=num_units, activation=activation, decay=0.1, epsilon=1e-20, alpha=0.5, optimize=True, optimize_vars=["rho", "decay", "alpha", "sw"], seed=random_seed)

inputA = Input(shape=(16,1024))
recurrent_layer = keras.layers.RNN(cell, return_sequences=True, name="ESN", dynamic=True)(inputA)
x1 = Dense(32, activation='relu')(recurrent_layer)
x1 = Flatten(name='fl1')(x1)

inputB = Input(shape=(1024))
x2 = Dense(32, activation='relu')(inputB)
x2 = Flatten(name='fl2')(x2)

z  = Concatenate()([x1,x2])
z = Dense(16, activation='relu')(z)
z = Dense(8, activation='relu')(z)
z = Dropout(0.1)(z)

output=Dense(nb_classes, activation='softmax')(z)

model = Model(inputs=[inputA,inputB], outputs=output)
print(model.summary())

############################### LOADING MODEL ################################
model_path = 'F:/Waseem_Data/Two_CNN_stream/UCF-Crime/ESN/Final/Model/fine_tune_model_two_steam_ESN_final.h5'
model.load_weights(model_path)
