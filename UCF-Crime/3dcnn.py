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
import cv2, pickle
from numpy import array
import scipy.io as sio

## Motion features :\
img_height , img_width = 224, 224
seq_len = 16
 
classes = ["Abuse", "Arrest","Arson","Assault","Burglary", "Explosion","Fighting","Normal_Videos_event", 
           "RoadAccidents","Robbery", "Shooting","Shoplifting", "Stealing",  "Vandalism"]
#Model
######## Optical flow
input_layer_2 = Input(shape=(16, 224, 224, 3))
Conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(input_layer_2)
Conv2 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(Conv1)
pool1 = MaxPooling3D(pool_size=(3, 3, 3))(Conv2)
dp1 = Dropout(0.25)(pool1)
pool2 = MaxPooling3D(pool_size=(3, 3, 3))(dp1)
dp2 = Dropout(0.25)(pool2)
fl_op = Flatten()(dp2)
dense_op = Dense(1024, activation='relu')(fl_op)
temporalmodel = Model(inputs=input_layer_2, outputs=dense_op)
temporalmodel.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(temporalmodel.summary())


#  Creating frames from videos
dataset_directory = r"E:/Waseem_Data/Two_CNN_stream/dataset/train"
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
                if frames_counter%17 == 16:
                    #print(frames_counter%17)
                    zm+=1
#                    frame = frame.reshape(1,30,224,224,3)
                    temp = np.asarray(video_features)
                    temp = temp.reshape(1, 16, 224, 224, 3)
                    feature = temporalmodel.predict(temp)

                    DatabaseFeautres.append(feature)
                    DatabaseLabel.append(dataset_folder[dir_counter])
                    video_features = []
                    frames_counter=0
 
print('Feature Extraction from dataset completed')
DatabaseFeautres = np.asarray(DatabaseFeautres)
print(DatabaseFeautres.shape)
TotalFeatures= []
HotArray = []

TotalFeatures = np.asarray(DatabaseFeautres)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres),1024])

OneHotArray = []
kk=1;
for i in range(len(DatabaseFeautres)-1):
    OneHotArray.append(kk)
    if (DatabaseLabel[i] != DatabaseLabel[i+1]):
        kk=kk+1;

with open("OneHotArray.pickle", 'wb') as f:
  pickle.dump(OneHotArray, f)
    
OneHot=  np.zeros([len(DatabaseFeautres),2], dtype='int');


for i in range(len(DatabaseFeautres)-1):
    print(i)
    OneHot[i,OneHotArray[i]-1] = 1



np.save('Train_3dd_features_2classes',TotalFeatures)
sio.savemat('Train_3dd_labels_2classes.mat', mdict={'Train_3dd_labels': OneHot})






























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




######## Optical flow
input_layer_2 = Input(shape=(15, 224, 224, 3))
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





