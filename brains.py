import sys
import math
import os
import pickle
from os import path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


join = lambda n: path.join('IPC', n)

def scale_and_split_data(data, inp_size, op_size=1, train_data_percent=0.8):
    data = data.filter(['close'])
    dataset = data.to_numpy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)
    train_data_len = math.ceil(len(dataset) * train_data_percent)

    train_data = scaled_data[0: train_data_len, :]

    x_train = []
    y_train = []

    for i in range(inp_size, len(train_data)):
        if len(train_data[i:i+op_size, 0]) == op_size:
            x_train.append(train_data[i-inp_size:i, 0])
            y_train.append(train_data[i:i+op_size, 0])
    
    test_data = scaled_data[train_data_len:]
    x_test = []
    y_test = []

    for i in range(inp_size, len(test_data)):
        if len(test_data[i:i+op_size, 0]) == op_size:
            x_test.append(test_data[i-inp_size:i, 0])
            y_test.append(test_data[i:i+op_size, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    return (x_train, y_train), (x_test, y_test), scaler


def create_model(inp_size, op_size, no_lstm_layers=1, no_lstm_neurons=10,
    no_dense_layers=1, no_dense_neurons=10, optimizer='adam', 
    loss='mean_squared_error'):
    model = Sequential()
    model.add(
        LSTM(
            no_lstm_neurons, 
            return_sequences=True,
            input_shape=(inp_size, 1)
        ) 
    )
    
    for _ in range(no_lstm_layers-2):
        model.add(
            LSTM(no_lstm_neurons, 
                return_sequences=True,
                # input_shape=(inp_size, 1)
            ) 
        )
    
    model.add(
        LSTM(
            no_lstm_neurons, 
            return_sequences=False,
            # input_shape=(inp_size, 1)
        ) 
    )
    
    
    for _ in range(no_dense_layers-1):
        model.add(Dense(no_dense_neurons))

    model.add(Dense(op_size))

    model.compile(optimizer=optimizer, loss=loss,  metrics=['accuracy'])
    return model


def predict(model, scaler, data, interval_pd_format):
    data = data.filter(['close'])
    dataset = data.to_numpy()
    
    scaled_data = scaler.fit_transform(dataset)
    scaled_data = np.array(scaled_data, ndmin=2)
    print("shape of scaled data: ", scaled_data.shape)
    prediction = model.predict(scaled_data)
    prediction = scaler.inverse_transform(prediction)
    # print(prediction.shape)
    prediction = prediction[0]

    pred_df = pd.DataFrame(
        data=prediction,
        columns=['prediction'],
        index=pd.date_range(
            start=data.index[-1],
            periods=len(prediction),
            freq=interval_pd_format
        )
    )
    data = pd.concat([data, pred_df])
    return data


def rmse(model, scaler, x_test, y_test):
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    
    return np.sqrt(
        np.mean(
            (prediction - y_test)**2
        )
    )

def load_model(file_name):
    return models.load_model(file_name)


if __name__ == '__main__':
    # pass
    if len(sys.argv) == 3:
        x_train = np.load(
            file=join('x_train.npy'),
            allow_pickle=False
        )

        y_train = np.load(
            file=join('y_train.npy'),
            allow_pickle=False
        )

        model = models.load_model(join('model.h5'))
        history = model.fit(
            x_train,
            y_train,
            batch_size=int(sys.argv[2]),
            epochs=int(sys.argv[1])
        )

        model.save(join('model.h5'))
        with open(join('history'), 'wb') as hf:
            pickle.dump(history.history, hf)


    # model = create_model( 50, 2, 2, 50, 2, 25 )
    # model.save(join('model.h5'))

    # from miner import get_historical_price
    # df = get_historical_price('AAPL', 10000, '15min')
    # df = pd.read_csv('df.csv')
    
    # train, test, scaler = scale_and_split_data(df, 60, 2)
    # print(f'type = {type(train[0])}, shape = {train[0].shape}')
    # with open('train_test.txt', 'w') as file:
    #     print(f'{train}\n\n{test}', file=file)

    # print(f'{train[0][0]}, {train[1][0]}\n\n{test[0][0]}, {test[1][0]}')



