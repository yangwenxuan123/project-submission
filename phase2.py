import numpy as np
import pandas as pd
import pickle

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import model_from_json

training = pd.read_pickle('test_2.pkl')

activelist = {"1" : np.array([0,1]), 
              "2" : np.array([2,3]), 
              "3" : np.array([4,5]), 
              "4" : np.array([6,7]), 
              "5" : np.array([8,9]), 
              "6" : np.array([10,11]), 
              "7" : np.array([12,13]), 
              "8" : np.array([14,15]), 
              "9" : np.array([16,17]), 
              "10" : np.array([18,19]), 
              "11" : np.array([20,21])
              }

nonactivelist = {"1" : np.array([0,1]),
                 "2" : np.array([2,3]), 
                 "3" : np.array([4,5]),
                 "4" : np.array([6,7]),
                 "5" : np.array([8,9]),
                 "6" : np.array([10,11]),
                 "7" : np.array([12,13]),
                 "8" : np.array([14,15]),
                 "9" : np.array([16,17]),
                 "10" : np.array([18,19]),
                 "11" : np.array([20,21]),
                 "12" : np.array([22,23]),
                 "13" : np.array([24,25]),
                 "14" : np.array([26,27]),
                 "15" : np.array([28,29]),
                 "16" : np.array([30,31]),
                 "17" : np.array([32,33]),
                 "18" : np.array([34,35]),
                 "19" : np.array([36,37]),
                 "20" : np.array([38,39]),
                 "21" : np.array([40,41]),
                 "22" : np.array([42,43])      
                 }


def build_rnn(X_train,y_train,regressor):
    regressor.add(LSTM(units = 512, return_sequences = True,input_shape = (X_train.shape[1], 390)))
    regressor.add(LSTM(units = 512, return_sequences=True))
    regressor.add(Flatten())
    regressor.add(Dense(units = 2))
    regressor.compile(optimizer = 'adagrad', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train,  epochs = 100 ,batch_size = 50)
    
     
predicted = {}
for player in range(1,2):#len(activelist)
    print("Active PLayer:",player)
    playerX = activelist[str(player)][0]
    playerY = activelist[str(player)][1]
    sq = "sequence_"
    for i in range(1,len(training)):
        predicted_per_sq = np.array([])
        regressor = Sequential()
        sq = sq + str(i)
        print("Sequence number----->",sq)
        train = training[(str(player),sq)]
        X_train = []
        y_train = []
        for i in range(50, len(train)):
            X_train.append(train[i-50:i, 0:390])
            y_train.append(train[i, playerX:playerY+1])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 390))
        build_rnn(X_train,y_train,regressor)
        predicted_per_sq = regressor.predict(X_train)
        predicted.update({(str(player),sq):predicted_per_sq})        
        sq = "sequence_"
file = open('predicted.pkl','wb')
pickle.dump(predicted,file)
file.close() 


#json_file = open("player_1.json", 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("player_1.h5")
#print("Loaded model from disk")
#predictedModel = loaded_model.predict(X_train)

sq = "sequence_"
dataset_test = pd.read_pickle('test_data_1.pkl') #STATS DATASET
test_set_1 = pd.read_pickle('test_1.pkl')#xtrain and ytrain combined
for player in range(1,2):#len(activelist)+1):
        
    playerX = activelist[str(player)][0]
    playerY = activelist[str(player)][1]
    for i in range(1,2):
        sq = sq + str(i)
        ground_truth_position = dataset_test[sq][:,playerX:playerY+1]
        test = test_set_1[(str(player),sq)]
        X_test = []
        for i in range(50, len(test)):
            X_test.append(test[i-50:i, 0:390])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 390))
    sq = 'sequence_'
predicted = regressor.predict(X_train)
