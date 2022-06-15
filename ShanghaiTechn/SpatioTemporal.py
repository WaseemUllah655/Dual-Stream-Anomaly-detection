import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils import Sequence, to_categorical
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.models import Model, Sequential
from keras.layers import *
from keras.applications.vgg16 import VGG16
#from keras.applications.resnet import  ResNet101
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import keras
import os
import cv2
from numpy import array
### EfficientNet
X = np.vstack(Train_features_efn_14_classes_b7)
print(X.shape)
X=X.reshape(X.shape[0],16,1024)
#X=X.reshape((X.shape[0], X.shape[1],1))

#X = np.moveaxis(X,[1,-1],[-1,1])
#X_test = np.moveaxis(X_test,[1,-1],[-1,1])

#y_test =DatabaseLabel000
#y_train = DatabaseLabel
y = TestLabels


X_train_Eff, X_val_Eff, y_train_Eff, y_val_Eff = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)

## Motion features :\
img_height , img_width = 224, 224
seq_len = 16
 
classes = ["Abuse", "Arrest","Arson","Assault","Burglary", "Explosion","Fighting","Normal_Videos_event", 
           "RoadAccidents","Robbery", "Shooting","Shoplifting", "Stealing",  "Vandalism"]
 
#  Creating frames from videos
dataset_directory = r"D:\Waseem_Data\waseem_2nd_paper\dataset\train"
dataset_folder = os.listdir(dataset_directory)

DatabaseFeautres = []
DatabaseLabel = []
cc=0

for dir_counter in range(0,len(dataset_folder)):
    cc+=1
    
    single_class_dir = dataset_directory + "/" + dataset_folder[dir_counter]
    all_videos_one_class = os.listdir(single_class_dir)
    print ('***********Processing', dataset_folder[dir_counter])
    
    for single_video_name in all_videos_one_class:
        #print(cc)
        video_path = single_class_dir + "/" + single_video_name
        #print ('Feature extracting: ', video_path)
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_features = []
        frames_counter = 0
        zm=0
        #print('=======================================================')
        while(frames_counter < total_frames-1):

            frames_counter+=1
            ret, frame = capture.read()
            if (ret):
                frame = cv2.resize(frame, (224,224))
#                frame = frame.reshape(1,224,224,3)
                video_features.append(frame)
                if frames_counter%16 == 15:
                    zm+=1
#                    frame = frame.reshape(1,30,224,224,3)
                    temp = np.asarray(video_features)
                    DatabaseFeautres.append(temp)
                    DatabaseLabel.append(dataset_folder[dir_counter])
                    video_features = []
                    frames_counter=0
 
DatabaseFeautres = np.asarray(DatabaseFeautres)
DatabaseLabel = np.asarray(DatabaseLabel)

OneHotArray = []
kk=1;
for i in range(len(DatabaseFeautres)-1):
    OneHotArray.append(kk)
    if (DatabaseLabel[i] != DatabaseLabel[i+1]):
        kk=kk+1;

OneHot=  np.zeros([len(DatabaseFeautres),2], dtype='int');
for i in range(len(DatabaseFeautres)-1):
    print(i)
    OneHot[i,OneHotArray[i]-1] = 1


X_train_op, X_test_op, y_train_op, y_test_op = train_test_split(DatabaseFeautres, OneHot, test_size=0.20, shuffle=False, random_state=0)



######## EfficientNet
input_layer_1 = Input(shape=(30, 1000))
lstm1 = GRU(256, activation='relu', dropout=0.2, return_sequences=True)(input_layer_1)
lstm2 = GRU(256, activation='relu',dropout=0.2, return_sequences=True)(lstm1)
output_block_1 = keras.layers.add([lstm2, lstm1])
lstm3 = GRU(256, activation='relu',dropout=0.2, return_sequences=False)(output_block_1)
dense_eff = Dense(128, activation='relu')(lstm3)

#Spatial model
Spatialmodel = Dense(2, activation='softmax')(dense_eff)
Spatialmodel = Model(inputs=input_layer_1, outputs=Spatialmodel)
Spatialmodel.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(Spatialmodel.summary())




######## Optical flow
input_layer_2 = Input(shape=(30, 224, 224, 3))
Conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(input_layer_2)
Conv2 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(Conv1)
pool1 = MaxPooling3D(pool_size=(3, 3, 3))(Conv2)
dp1 = Dropout(0.25)(pool1)
pool2 = MaxPooling3D(pool_size=(3, 3, 3))(dp1)
dp2 = Dropout(0.25)(pool2)
fl_op = Flatten()(dp2)
dense_op = Dense(128, activation='relu')(fl_op)

##temporal model
temporalmodel = Dense(2, activation='softmax')(dense_op)
temporalmodel = Model(inputs=input_layer_2, outputs=temporalmodel)
temporalmodel.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(Spatialmodel.summary())



merg_x = add([dense_eff, dense_op])



#con = Add()([dense_eff, dense_op])
con = Flatten()(merg_x)
dense_1 = Dense(64, activation='sigmoid')(merg_x)
dense_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(64, activation='sigmoid')(dense_1)
dense_2 = Dropout(0.5)(dense_2)
final = Dense(2, activation='softmax')(dense_2)

#SpatioTemporalmodel
SpatioTemporalmodel = Model(inputs=[input_layer_1, input_layer_2], outputs=final)
SpatioTemporalmodel.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(SpatioTemporalmodel.summary())
#X_train_Eff, X_val_Eff, y_train_Eff, y_val_Eff, X_train_op, X_test_op, y_train_op, y_test_op



sptmModel=SpatioTemporalmodel.fit([X_train_Eff, X_train_op ],array(y_train_op), epochs=5, batch_size = 4, validation_split=0.2)


Spatialmodel=Spatialmodel.fit(X_train_Eff,y_train_Eff, epochs=5, batch_size = 1, validation_split=0.2)
temporalmodel=temporalmodel.fit(X_train_op,y_train_op, epochs=5, batch_size = 1, validation_split=0.2)
#models summaries
SpatioTemporalmodel.save('SpatioTemporalmodel.h5')
Spatialmodel.save('Spatialmodel.h5')
temporalmodel.save('temporalmodel.h5')

Spatialmodel.summary()
temporalmodel.summary()
SpatioTemporalmodel.summary()

pred=SpatioTemporalmodel.predict([X_val_Eff,X_test_op], y_val_Eff)





