# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 00:25:50 2022

@author: Waseem_Ullah
"""

from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.ops import math_ops
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.optimizers import SGD
from ESN import EchoStateRNNCell
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense, Activation, MaxPooling2D,Conv1D, Dropout,MaxPooling1D, Bidirectional,LSTM, GRU, Flatten, Concatenate, BatchNormalization
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
from scipy.io import loadmat
def plot_confusion_matrics(cm, classes, normalize=False, title='Confusion Matrics', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm=cm.astype('float')/cm.sum(axis-1)[:, np.newaxis]
		print('normalized cm')
	else:
		print('without normalization')
	print(cm)
	thresh=cm.max()/2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i,j]>thresh else 'black')
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
    

 
nb_classes = 2
batch_size = 128
epochs = 200
X_train=np.load(r'D:/2D_3D two streem final/Violance_datasets_two_stream_model/Two_CNN_stream/ShanghaiTechn/efficientNet_features/Train_Shanghaitech_refine_features_efn_2_classes_b7.npy')
X_train = np.vstack(X_train)
#X_train=TrainVIT16_15Frames_244

X_test=np.load(r'D:/2D_3D two streem final/Violance_datasets_two_stream_model/Two_CNN_stream/ShanghaiTechn/efficientNet_features\Test_Shanghaitech_refine_features_efn_2_classes_b7.npy')
X_test = np.vstack(X_test)

X_train=X_train.reshape(X_train.shape[0],16,1024)
X_test=X_test.reshape(X_test.shape[0],16,1024)   
    
y_train = loadmat(r'D:/2D_3D two streem final/Violance_datasets_two_stream_model/Two_CNN_stream/ShanghaiTechn/efficientNet_features/Train_Shanghaitech_refine_labels_efn_2_classes_b7.mat')
y_train=y_train['Total_labeL']
y_train=TestLabels_000
y_test = loadmat(r'D:/2D_3D two streem final/Violance_datasets_two_stream_model/Two_CNN_stream/ShanghaiTechn/efficientNet_features\Test_Shanghaitech_refine_labels_efn_2_classes_b7.Mat')
y_test=y_test['Total_labeL']
y_test=TestLabels

X_train,X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.20,random_state=42)
#X_train,X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.30,random_state=42)


clear_session()
batches = 20
stime = 16
epochs = 400
num_units = 64
num_inputs = 1024
num_outputs = 2
   
activation = lambda x: math_ops.tanh(x)

#input_layer = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
cell = EchoStateRNNCell(units=num_units, 
                           activation=activation, 
                           decay=0.1, 
                           epsilon=1e-20,
                           alpha=0.5,
                           optimize=True,
                           optimize_vars=["rho", "decay", "alpha", "sw"],
                           seed=random_seed)
recurrent_layer = keras.layers.RNN(cell, input_shape=(16,1024), 
                                   return_sequences=True, name="nn")

#output = keras.layers.Dense(num_outputs, name="readouts")
input_layer = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

model = recurrent_layer(input_layer)
model = BatchNormalization()(model)
model = BatchNormalization()(model)
model = Dropout(0.3)(model)
model = Flatten()(model)
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model = Dense(128, activation='relu')(model)
model = Dropout(0.2)(model)
model = Dense(64, activation='relu')(model)
model = Dropout(0.1)(model)
out = Dense(2, activation='softmax')(model)


model = Model(inputs=input_layer, outputs=out)
print(model.summary())


sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer="Nadam", metrics=['accuracy'])
#history=model.fit(X, y_train, epochs=20,valdation_split=0.2, batch_size=100, verbose=1)
checkpoint_filepath = 'Model_weights/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
# history1=model.fit(X_train,y_train,batch_size = 128, epochs=100, validation_data=(X_val,y_val), verbose=1)
history1=model.fit(X_train,y_train,batch_size =128,epochs=200, callbacks=[model_checkpoint_callback], validation_data=(X_val,y_test), verbose=1)

#model.save_weights('weights/Zulfi')
model.load_weights('weights/')
pred=model.predict(X_test)
score, acc = model.evaluate(X_test,y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

y_pred = (pred > 0.5)*1 
confusion = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
import matplotlib.pyplot as plt

Y_S = y_test
PreLabels =  y_pred
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc_1 = dict()
n_classes = 2

h = Y_S.shape[0]
w = Y_S.shape[1]
ns_probs = np.zeros((h,w),int)

ns_auc = roc_auc_score(np.array(Y_S), np.array(ns_probs), multi_class="ovr")

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_S[:, i], PreLabels[:, i])
    roc_auc_1[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_S.ravel(), PreLabels.ravel())
roc_auc_1["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(15,10))
lw = 2
plt.plot(fpr[1], tpr[1], color='royalblue', lw=lw, label='Logistic: ROC AUC=%.3f' % ( roc_auc_1[1]))
plt.plot([0, 1], [0, 1], color='maroon', lw=lw, linestyle='--', label='No Skill: ROC AUC=%.3f' % (ns_auc))
plt.grid(linestyle='-', linewidth='0.3', color='red') # for grid
plt.axhline(y=0, color='k') # X-axis Line
plt.axvline(x=0, color='k') # Y-axis Line
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontweight = 'bold')
plt.ylabel('True Positive Rate', fontweight = 'bold')
plt.title('Receiver operating characteristic example')
plt.legend(loc='center right')
plt.savefig("ESN_Efficient.jpg")
plt.show()
