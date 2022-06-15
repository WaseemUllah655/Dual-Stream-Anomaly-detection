import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import sys, os
import numpy as np
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
import tensorflow.keras as keras
from tensorflow.python.ops import math_ops
from ESN import EchoStateRNNCell
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras import backend
from math import sqrt
from tensorflow.keras.layers import Flatten
import tensorflow as tf
import mbe
from keras.utils.vis_utils import plot_model
import visualkeras
import pydot
# memory growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
random_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print("seed: ", random_seed)

def nrmse(y_true, y_pred):
    """ Normalized Root Mean Squared Error """
    nrmsee= (sqrt(sqrt(mean_squared_error(y_true, y_pred) / (y_true.max() - y_true.min()))))
    return nrmsee

def MBEE(y_true, y_pred):
    MBE = np.mean(y_true - y_pred) #here we calculate MBE
    #NMBE=MBE*100
    return MBE

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))) 



def split(data, pred, max_values):
    Y = []
    X=[]
    cnt=max_values/10
    cnt=int(cnt)
    c=0
    for i in range (cnt):
        
        X.append(data[c:c+10])
        Y.append(pred[c:c+10])
        c+=1

    return X,Y

max_values=1048560


###################training    

df = pd.read_csv('78-Site_2-eco-Kinetics.csv', sep=',', 
                 parse_dates={'dt' : ['timestamp']}, infer_datetime_format=True, 
                 low_memory=False, index_col='dt', 
                 usecols = ['timestamp', 'Active_Energy_Delivered_Received', "Current_Phase_Average",
                            "Active_Power", "Wind_Speed", "Weather_Temperature_Celsius", "Weather_Relative_Humidity", 
                            "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation", "Wind_Direction", "Weather_Daily_Rainfall"  ])

df_p = pd.read_csv('78-Site_2-eco-Kinetics.csv', sep=',', 
                 parse_dates={'dt' : ['timestamp']}, infer_datetime_format=True, 
                 low_memory=False,  index_col='dt', 
                 usecols = ['timestamp', "Active_Power"])

        
df=df.interpolate(method='linear')
df=df.fillna(1)

df_p=df_p.interpolate(method='linear')
df_p=df_p.fillna(1)




trainvalues = df.values
predvalues = df_p.values
scaler = preprocessing.MinMaxScaler(feature_range=(0,2))
trainvalues = scaler.fit_transform(trainvalues)
predvalues = scaler.fit_transform(predvalues)

print(predvalues)


X, Y=split(trainvalues,predvalues, max_values)
X=array(X)
Y=array(Y)
print(X.shape)

X, valx, Y, Valy=train_test_split(X, Y, test_size=0.3, random_state=random_seed)

batches = 20
stime = 10
epochs = 400
num_units = 16
num_inputs = 10
num_outputs = 10
   
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
recurrent_layer = keras.layers.RNN(cell, input_shape=(stime, num_inputs), 
                                   return_sequences=True, name="nn", dynamic=True)
recurrent_layer2 = keras.layers.RNN(cell, return_sequences=True, name="nn2", dynamic=True)
output = keras.layers.Dense(num_outputs, name="readouts")
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model = keras.models.Sequential()
model.add(recurrent_layer)
#model.add(recurrent_layer2)
model.add(Conv1D(filters=8,kernel_size=1,  activation='relu'))
model.add(Conv1D(filters=16,kernel_size=1,  activation='relu'))
model.add(Conv1D(filters=32,kernel_size=1,  activation='relu'))
model.add(Flatten())
model.add(output)
model.compile(loss="mse", optimizer=optimizer)
model.summary()



hist = model.fit(X, Y, epochs=50, verbose=1, batch_size=500, validation_split=0.2)
visualkeras.layered_view(model)


pred=model.predict(valx,verbose=1)
Valy=Valy.reshape(Valy.shape[0], Valy.shape[1])
pred=pred.flatten()
Valy=Valy.flatten()


plt.figure(figsize=(6, 3))
plt.plot(Valy[0:240],  marker='.', linewidth=3,markersize=2, color='royalblue', label='orignal')
plt.plot(pred[0:240],  marker='*', linewidth=3,markersize=2, color='darkorange', label='predicted')
plt.xlabel(' Hourly consumption ', fontsize=11, color='black', style='normal', fontname="Time New Roman")
plt.ylabel('Electricity consumption (KWh)', fontsize=10, color='black', style='normal', fontname="Time New Roman")
plt.xticks(fontsize=10, color='black', style='normal', fontname="Time New Roman", ha="center")
plt.yticks(fontsize=10, color='black', style='normal', fontname="Time New Roman", )
plt.grid(True)
plt.legend()
plt.show()





print('MSE:   ', mean_squared_error(Valy,pred))
print('MAE:   ', mean_absolute_error(Valy,pred))
print('RMSE:  ', sqrt(mean_squared_error(Valy,pred)))
print('MAPE:  ', MAPE(Valy,pred))
print('MBE;   ', MBEE(Valy,pred))
print('NRMSE: ', nrmse(Valy,pred))


mbee=MBEE(Valy,pred)
print(mbee)