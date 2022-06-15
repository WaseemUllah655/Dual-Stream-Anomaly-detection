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
from tensorflow.keras.callbacks import EarlyStopping
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
epochs = 20
X_vgg = np.vstack(HockeyFights_Arrange_features_efn_2_classes_b7)
#X_test = np.vstack(Test_features_efn_2_classes_b7)


y_train = TestLabels
#y_test =TestLabels

train_features_final_15frames = np.vstack(HockeyFights_Arrange_3dd_features_2classes)
#test_features_final=np.vstack(Test_3dd_features_2classes)

X_vgg=X_vgg.reshape(X_vgg.shape[0],16,1024)
#X_test=X_test.reshape(X_test.shape[0],16,1024)

train_features_final_15frames=train_features_final_15frames.reshape(train_features_final_15frames.shape[0],1024)
#test_features_final=test_features_final.reshape(test_features_final.shape[0],1024)

y_train_pf= HockeyFights_Arrange_3dd_labels_2classes
#y_test_pf =Test_3dd_labels_2classes

X_train_OP,X_val_OP, y_train_OP, y_val_OP = train_test_split(X_vgg,y_train,test_size=0.2,random_state=42)

X_train_OP,X_test_OP, y_train_OP, y_test_OP = train_test_split(X_train_OP,y_train_OP,test_size=0.2,random_state=42)


X_train_I3D,X_val_I3D, y_train_I3D, y_val_I3D = train_test_split(train_features_final_15frames,y_train_pf,test_size=0.2,random_state=42)

X_train_I3D,X_test_I3D, y_train_I3D, y_test_I3D = train_test_split(X_train_I3D,y_train_I3D,test_size=0.2,random_state=42)

# history=model.fit([X_train_OP,X_train_I3D] ,y_train_OP,batch_size=batch_size, epochs=epochs,validation_data=([X_val_OP,X_val_I3D], y_val_OP))
# summarize layers
clear_session()
batches = 16
stime = 16
epochs = 400
num_units = 32
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

x2 = Dense(128, activation='relu')(inputB)
x2 = Dense(64, activation='relu')(inputB)
x2 = Dense(32, activation='relu')(inputB)
x2 = Flatten(name='fl2')(x2)
'''
x2 = Dense(256, activation='relu')(x2)
x2 = Dense(128, activation='relu')(x2)
x2 = Dense(64, activation='relu')(x2)'''
z  = Concatenate()([x1,x2])

z = Dense(64, activation='relu')(z)
z = Dense(32, activation='relu')(z)
z = Dense(8, activation='relu')(z)
z = Dropout(0.1)(z)
#z = Dense(1024,activation='relu')(z)

#z = Dense(128, activation='relu')(z)

output=Dense(nb_classes, activation='softmax')(z)

model = Model(inputs=[inputA,inputB], outputs=output)
#print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# custom_early_stopping = EarlyStopping(
#     monitor='val_accuracy', 
#     patience=10, 
#     min_delta=0.001, 
#     mode='max'
# )

history=model.fit([X_train_OP,X_train_I3D] ,y_train_OP,batch_size=128,epochs=200,validation_data=([X_val_OP,X_val_I3D], y_val_OP))
score, acc = model.evaluate([X_val_OP,X_val_I3D],y_val_OP, batch_size=batch_size)




Eff_Features = Features
Eff_Features=Eff_Features.reshape(Eff_Features.shape[0],16,1024)

I3D_Features = Dfeatures

import cv2
from tensorflow.keras.models import load_model
import time


nb_classes =  ['Normal', 'Anomaly']

filename = '01_0014.avi'
#VideoName = patth[len(patth)-1]


font                   = cv2.FONT_HERSHEY_TRIPLEX
bottomLeftCornerOfText = (10,10)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1


#Read input video
capture = cv2.VideoCapture(filename)

#Get FPS which is used to saved the output video
fps = capture.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('01_0014_results.avi',fourcc, fps, (224,224))

#Video processing and output generation
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames==0:
    print('your video file are not loading')

    
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
                op = Eff_Features[c2,:].reshape(1,16,1024)
                vgg = I3D_Features[c2,:].reshape(1,1024)
                start = time.time()
                score = model.predict([op,vgg]);  
                end = time.time()
                seconds = end - start
                fps  = 1.0 / seconds
                
                zm=seconds/16
                print("FPS", zm*25)

                print("Estimated frames per second : {0}".format(fps))
                #print ("Time taken : {0} seconds".format(seconds))
    # Number of frames to capture

                # num_frames = 120;

                # print("Capturing {0} frames".format(num_frames))

                # # # Start time
                # # start = time.time()
                # for i in range(0, num_frames) :

                #     ret, frame = capture.read()

 

                # end = time.time()

                # Time elapsed
                # seconds = end - start

                # print ("Time taken : {0} seconds".format(seconds))
 

                # # Calculate frames per second

                # fps  = num_frames / seconds

                # print("Estimated frames per second : {0}".format(fps))

                label = score.argmax(axis=1)
                confid = score[0,label[0]]
                Category = nb_classes[label[0]]
                Text = 'Category: '+ Category +'   Confidence: '+ str(confid)
                print(Text)
            for j in frames:
                cv2.putText(j, Text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 255, 255), 1, cv2.LINE_4) 
                out.write(j)
                frames = []

out.release()
capture.release()

