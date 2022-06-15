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

nb_classes = 2
batch_size = 128
epochs = 20
X_vgg = np.vstack(Train_features_efn_b7_classes_2)
X_test = np.vstack(Test_features_efn_2_classes_b7)


y_train = TrainLabels_2Classes
y_test =TestLabels

train_features_final_15frames = np.vstack(Train_3dd_features_2classes)
test_features_final=np.vstack(Test_3dd_features_2classes)

X_vgg=X_vgg.reshape(X_vgg.shape[0],16,1024)
X_test=X_test.reshape(X_test.shape[0],16,1024)

train_features_final_15frames=train_features_final_15frames.reshape(train_features_final_15frames.shape[0],1024)
test_features_final=test_features_final.reshape(test_features_final.shape[0],1024)

y_train_pf= Train_3dd_labels_2classes
y_test_pf =Test_3dd_labels_2classes

X_train_OP,X_val_OP, y_train_OP, y_val_OP = train_test_split(X_vgg,y_train,test_size=0.2,random_state=42)

X_train_I3D,X_val_I3D, y_train_I3D, y_val_I3D = train_test_split(train_features_final_15frames,y_train_pf,test_size=0.2,random_state=42)

# history=model.fit([X_train_OP,X_train_I3D] ,y_train_OP,batch_size=batch_size, epochs=epochs,validation_data=([X_val_OP,X_val_I3D], y_val_OP))
# summarize layers
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
inputA = Input(shape=(16,1024))
recurrent_layer = keras.layers.RNN(cell, return_sequences=True, name="ESN", dynamic=True)(inputA)
#recurrent_layer = keras.layers.RNN(cell, return_sequences=True, name="ESN", dynamic=True)(inputA)
##recurrent_layer = keras.layers.RNN(cell, return_sequences=True, name="ESN", dynamic=True)(inputA)

inputB = Input(shape=(1024))
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
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history=model.fit([X_train_OP,X_train_I3D] ,y_train_OP,batch_size=256, epochs=20,validation_data=([X_val_OP,X_val_I3D], y_val_OP))
score, acc = model.evaluate([X_test,test_features_final],y_test, batch_size=batch_size)

# summarize layers

# plot graph
#plot_model(model, to_file='multiple_inputs.png')
model.save('fine_tune_model_two_steam_ESN.h5')
model.save_weights('fine_tune_model_two_steam_ESN_final.h5')


from keras.models import load_model
model.load_weights('1.h5')


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)






score, acc = model.evaluate([X_test,test_features_final],y_test, batch_size=batch_size)

pred= model.predict([X_test,test_features_final])


#pred= model.predict([X_val_OP,X_val_I3D])



print(history.history)
history = history
plt.plot()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(' model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.savefig('training.jpg')
plt.show()
#
y_pred = (pred > 0.5)*1 
confusion = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(' model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.jpg')
plt.show()
import numpy as np
accy = history.history['accuracy']
np_accy = np.array(accy)
#np.savetxt('save_acc.txt', np_accy)
model.save('Two-stream-CNN.h5')
score, acc = model.evaluate([X_val_OP,X_val_I3D], y_val_OP, batch_size=128)
score, acc = model.evaluate([X_test,test_features_final],y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

############################################################
############################################################
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# generate a no skill prediction (majority class)
w = y_test.shape[1]
h = y_test.shape[0]

ns_probs = np.zeros((h,w), dtype=int)
# keep probabilities for the positive outcome only
lr_probs = y_pred 

# calculate scores
ns_auc_1 = roc_auc_score(y_test, ns_probs)
lr_auc_1 = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc_1))
print('Logistic: ROC AUC=%.3f' % (lr_auc_1))


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc_1 = dict()
n_classes = 2

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
#     roc_auc_1[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
# roc_auc_1["micro"] = auc(fpr["micro"], tpr["micro"])

# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_1[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

confusion = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1))

target_names = ['Normal', 'Anomaly']

print('Classification Report')
print(classification_report(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1), target_names=target_names))

############################____________NEWWWWWWWW___CODE
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
plt.savefig("ROC_Waseem_1.jpg")
plt.show()




import seaborn as sns

df = sns.heatmap(confusion, annot=True)
#df.figure.savefig("D:/Umair_Haroon/Behaviour_Recognition/3DCNN's/New folder (2)/NEW/image.png")

con = np.zeros((n_classes,n_classes))
for x in range(n_classes):
    for y in range(n_classes):
        con[x,y] = confusion[x,y]/np.sum(confusion[x,:])
        
print(con)
# sns_plot = sns.pairplot(df, hue='species', size=2.5)

df = sns.heatmap(con, annot=True,fmt='.2%', cmap='Blues',xticklabels= target_names , yticklabels= target_names)
df.figure.savefig("CM_image2.png")