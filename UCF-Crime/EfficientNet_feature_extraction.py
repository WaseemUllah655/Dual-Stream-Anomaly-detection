import numpy
#import matplotlib.pyplot as plt

from keras.models import Model

import tensorflow as tf
#from keras.applications.mobilenetv2 import MobileNetV2
 
import matplotlib.pyplot as plt

from keras.layers import Dense, Flatten

import cv2

import pickle

import numpy as np

import scipy.io as sio

from keras.preprocessing import image

from keras.applications.vgg19 import preprocess_input

import os 
import efficientnet.keras as efn
import itertools

# def plot_confusion_matrics(cm, classes, normalize=False, title='Confusion Matrics', cmap=plt.cm.Blues):
# 	plt.imshow(cm, interpolation='nearest', cmap=cmap)
# 	plt.title(title)
# 	plt.colorbar()
# 	tick_marks=np.arange(len(classes))
# 	plt.xticks(tick_marks, classes, rotation=45)
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

import tensorflow as tf
import keras




width = 600
height = 600
epochs = 20
NUM_TRAIN = 900
NUM_TEST = 152
dropout_rate = 0.2
input_shape = (height, width, 3)
eff = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
eff.summary()
x = eff.layers[-1].output
x=Flatten()(x)
x = Dense(1024, activation='relu')(x)

model = Model(input=eff.input,output=x)
model.summary()

dataset_directory = "dataset/test/"
dataset_folder = os.listdir(dataset_directory)

DatabaseFeautres = []
DatabaseLabel = []
cc=0
print('Feature Extraction from dataset ...')
cnt1=0
for dir_counter in range(0,len(dataset_folder)):
    cc+=1
    cnt1+=1
    print('Processing',cnt1 ,'of 2')
    
    single_class_dir = dataset_directory + "/" + dataset_folder[dir_counter]
    all_videos_one_class = os.listdir(single_class_dir)
    #print ('***********Processing', dataset_folder[dir_counter])
    
    for single_video_name in all_videos_one_class:
        print(cc)
        video_path = single_class_dir + "/" + single_video_name
        # print ('Feature extracting: ', video_path)
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_features = []
        


        frames_counter = -1
        zm=0
        #print('=======================================================')
        while(frames_counter < total_frames-1):

            frames_counter+=1
            ret, frame = capture.read()
            if (ret):
                frame = cv2.resize(frame, (600,600))
                img_data = image.img_to_array(frame)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                single_featurevector = model.predict(img_data)
                
                video_features.append(single_featurevector)
                #print('image features length : ' , (single_featurevector.shape))
                #print ('Shape = ' , single_featurevector.shape, " max = ", max(single_featurevector))
                #print(frames_counter%30)
                
                #if frames_counter%30==0:
                if frames_counter%16 == 15:
                    zm+=1
                    #print(zm)
                    #print(total_frames)
                    temp = np.asarray(video_features)
                    
                    DatabaseFeautres.append(temp)
                    DatabaseLabel.append(dataset_folder[dir_counter])
                    #print('extracted features len => ',len(DatabaseFeautres))
                    print('video features length : ',np.asarray(video_features).shape)
                    video_features = []

            #cv2.imshow('v', frame)
            #cv2.waitKey(2)

        #print (single_video + "\n")

print('Feature Extraction from dataset completed')
#print(np.asarray(DatabaseFeautres).shape)
TotalFeatures= []
HotArray = []
for sample in DatabaseFeautres:
    TotalFeatures.append(sample.reshape([1,16384]))


TotalFeatures = np.asarray(TotalFeatures)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres),16384])

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



np.save('Test_features_efn_2_classes_b7',TotalFeatures)
sio.savemat('Test_labels_efn_2_classes_b7.mat', mdict={'TestLabels': OneHot})


'''
print(TotalFeatures.shape)
print(OneHot.shape)


#########load data
X = np.vstack(mobilenetv2final)
print(X.shape)
y = DatabaseLabeltest
train_X = X.reshape((X.shape[0], 1, X.shape[1]))
print(train_X.shape)


########tran test split
X_train,X_val, y_train, y_val = train_test_split(train_X,y,test_size=0.4,random_state=42)
X_val,X_test, y_val, y_test = train_test_split(X_val,y_val,test_size=0.5,random_state=42)


###############moedl

input_layer = keras.layers.Input(shape=(train_X.shape[1], train_X.shape[2]))

lstm1=GRU(512, activation='relu', dropout=0.2, return_sequences=True)(input_layer)
lstm2=GRU(512, activation='relu',dropout=0.2, return_sequences=True)(lstm1)
output_block_1 = keras.layers.add([lstm2, lstm1])
lstm3=GRU(512, activation='relu',dropout=0.2, return_sequences=False)(output_block_1)
fc1=Dense(300, activation='relu')(lstm3)
fc2=Dense(64, activation='relu')(fc1)
model_out=Dense(5, activation='softmax')(fc2)
model = keras.models.Model(inputs=input_layer, outputs=model_out)
adm = Adam(lr=0.0001)
opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
history=model.fit(train_X, y, epochs=20, batch_size=100, validation_data=(X_val, y_val), verbose=1)

model.save('models/mobilenetv2_residuallstmFinal.h5')


##############test
pred=model.predict(X_test)


###############confusio metrics
y_pred = (pred > 0.5)*1 
confusion_matrix1=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


#cm=confusion_matrix(y_test, y_pred.argmax(axis=1))


cm_plot_labels=['Assault', 'Explosion','Fighting','Normal','RoadAccidents']
plot_confusion_matrics(confusion_matrix1, cm_plot_labels, title='Confusion Matrics')
#####################ploting

history = history
plt.plot()


print('==========================')
print('please enter the model name without spaces to save training, validation loss/acc and AUC graph')
name=input()
#plt.savefig('results/history.PNG')
# summarize history for acc
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('BiLSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results/'+name+'_tr_val_accuracy.jpg')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('BiLSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results/'+name+'_tr_val_loss.jpg')
plt.show()
#plt.savefig('results/loss.PNG')
import numpy as np
accy = history.history['acc']
np_accy = np.array(accy)
#np.savetxt('models/save_accmobilnet.txt', np_accy)
#model.save('bilstm_model_optical.h5')
score, acc = model.evaluate(X_test, y_test, batch_size=100)
print('Test score:', score)
print('Test accuracy:', acc)
print('############################################')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


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
n_classes = 5

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc_1[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc_1["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_1[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('results/'+name+'_tr_val_AUC.jpg')
plt.show()




N = np.arange(0, 20)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
#plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, history.history["acc"], label="train_acc")
#plt.plot(N, H.history["val_acc"], label="val_acc")
#plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

plt.savefig('4.jpg')'''


