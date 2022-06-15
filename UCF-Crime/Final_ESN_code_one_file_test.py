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
from tensorflow.keras.backend import clear_session
# Multiple Inputs
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
random_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print("seed: ", random_seed)

#Plotting confusion matrix
# def plot_confusion_matrics(cm, classes, normalize=False, title='Confusion Matrics', cmap=plt.cm.Blues):
# 	plt.imshow(cm, interpolation='nearest', cmap=cmap)
# 	plt.title(title)
# 	plt.colorbar()
# 	tick_marks=np.arange(len(classes))
# 	plt.xticks(tick_marks, classes, rotation=0)
# 	plt.yticks(tick_marks, classes)
# 	if normalize:
# 		cm=cm.astype('float')/cm.sum(axis-1)[:, np.newaxis]
# 		print('normalized cm')
# 	else:
# 		print('without normalization')
# 	print(cm)
# 	thresh=cm.max()/2.
# 	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# 		plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i,j]>thresh else 'black')
# 	plt.tight_layout()
# 	plt.ylabel('True label')
# 	plt.xlabel('Predicted label')
# 	plt.show()
    
Eff_Features = Features

I3D_Features = Dfeatures



nb_classes =  ['Normal', 'Anomaly']


clear_session()
batches = 16
stime = 16
epochs = 400
num_units = 16
num_inputs = 1024
num_outputs = 2
   
activation = lambda x: math_ops.tanh(x)


# Init the ESN cell
cell = EchoStateRNNCell(units=num_units, 
                        activation=activation, 
                        decay=0.1, 
                        epsilon=1e-20,
                        alpha=0.5,
                        optimize=True,
                        optimize_vars=["rho", "decay", "alpha", "sw"],
                        seed=random_seed)
inputA = Input(shape=(1024))
recurrent_layer = keras.layers.RNN(cell, return_sequences=True, name="ESN", dynamic=True)(inputA)
#recurrent_layer = keras.layers.RNN(cell, return_sequences=True, name="ESN", dynamic=True)(inputA)
##recurrent_layer = keras.layers.RNN(cell, return_sequences=True, name="ESN", dynamic=True)(inputA)

inputB = Input(shape=(16,1024))
#x1 = Bidirectional(GRU(256, return_sequences=True, name = 'LSTM_1'))(inputA)
#x1 = Bidirectional(GRU(256, return_sequences=False, name = 'LSTM_2'))(x1)
#
#x1 = Bidirectional(LSTM(128, return_sequences=False, name = 'LSTM_3'))(x1)
#x1 = Dropout(0.5)(x1)
#
x1 = Dense(32, activation='relu')(recurrent_layer)
'''x1 = Dense(256, activation='relu')(x1)
x1 = Dense(128, activation='relu')(x1)'''
x1 = Flatten(name='fl1')(x1)

x2 = Dense(32, activation='relu')(inputB)
x2 = Flatten(name='fl2')(x2)
'''
x2 = Dense(256, activation='relu')(x2)
x2 = Dense(128, activation='relu')(x2)
x2 = Dense(64, activation='relu')(x2)'''
z  = Concatenate()([x1,x2])

z = Dense(16, activation='relu')(z)
z = Dense(8, activation='relu')(z)
z = Dropout(0.1)(z)
#z = Dense(1024,activation='relu')(z)

#z = Dense(128, activation='relu')(z)

output=Dense(nb_classes, activation='softmax')(z)

model = Model(inputs=[inputA,inputB], outputs=output)
print(model.summary())


callbacks = [tboard_callback]

from profiler import model_profiler
%load_ext tensorboard
import cv2
from tensorflow.keras.models import load_model

aa=model.load_weights('Final_weights/')

filename = '01_0014.avi'
capture = cv2.VideoCapture(filename)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter("01_0014_results.avi", fourcc, 25.0, (320,240))

    
c=0
c1=0
c2=-1


frames = []
while (c < total_frames):
    c += 1
    (status_i, frame) = capture.read()
    
    if(status_i):
        c1 += 1
        frames.append(frame)
        if c1 % 16==15:
            c2 += 1
            if(c2 < len(Eff_Features)):
                op = I3D_Features
                vgg = Eff_Features[c2,:].reshape(1,16,1024)
                score = aa.predict([op,vgg]);  
                label = score.argmax(axis=1)
                confid = score[0,label[0]]
                Category = nb_classes[label[0]]
                Text = 'Category: '+ Category +'   Confidence: '+ str(confid)
                print(Text)
            for j in frames:
                cv2.putText(j, Text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 255, 255), 1, cv2.LINE_4) 
                out.write(j)
#                x=cv2.resize(j,(640,480))
                cv2.imshow('', x)
#                cv2.waitKey(1)
                frames = []

out.release()
capture.release()


