##importing the libraries
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle

data=pd.read_csv('dataset/Seethaeliya1.csv')
data['Date']=pd.to_datetime(data['Date'])

data2=data.filter(['RH_1'])
data2=data2.interpolate()

dataset2=data2.values

##train data set
train_data_len=math.ceil(len(dataset2)*.8)

##scaling the data
scaler=MinMaxScaler(feature_range=(0,1))
sc=scaler.fit_transform(dataset2)
#Create the training data set
##create the scaled training data set
train_data=sc[0:train_data_len,:]

##split the data into x_train and y_train data sets
x_train=[] ##features
y_train=[] ##target

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

##convert train sets to numpyarrays
x_train,y_train=np.array(x_train),np.array(y_train)
x_train
y_train

#reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

##Build The Model
model=Sequential()
model.add(LSTM(50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,activation='relu',return_sequences=False))
model.add(Dense(25,activation='relu'))
model.add(Dense(1))
#compile the model
model.compile(optimizer='adam',loss='mean_squared_error')

##train the model
history = model.fit(x_train,y_train,epochs=10,batch_size=1)

##create the testing dataset
##create a new array containing scaled values from index 1694 to 2192
test_data=sc[train_data_len-60:,:]
##Create data set x_test and y_test
x_test=[]
y_test=dataset2[train_data_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

##convert the data into numpy array
x_test=np.array(x_test)

##reshape the dataset
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

##get the models predicted temperatures
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions) ##unscaling

with open('models/rh_1.pkl', 'wb') as file:
    pickle.dump(model, file)