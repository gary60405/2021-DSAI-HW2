import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class Trader():
    def __init__(self, train_filename, test_filename, output_filename):
        self.model = None
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.output_filename = output_filename
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.stock = 0
        
    def model_building(self, input_size):
        model = Sequential()
        model.add(LSTM(units=900, return_sequences = True, kernel_initializer = 'glorot_uniform', input_shape  =  input_size))
        model.add(Dropout(0.3))
        model.add(LSTM(units = 900, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return model
        
    def data_loader(self):
        rankings_colname = [ 'open', 'high', 'low', 'close' ]
        train_data = pd.read_csv(self.train_filename, header=None, names = rankings_colname)
        test_data = pd.read_csv(self.test_filename, header=None, names = rankings_colname)
        return train_data, test_data
        
    
    def data_preprocessing(self):
        #擷取open欄位的data
        train_open = self.train_data.iloc[:, 0:1].values

        #正規化
        train_open_scaled = self.scaler.fit_transform(train_open)
        # print(train_open_scaled)

        # Feature selection
        xtrain = []
        ytrain = []
        for i in range(1, len(train_open_scaled)):
            xtrain.append(train_open_scaled[i - 1 : i, 0])
            ytrain.append(train_open_scaled[i, 0])

        xtrain, ytrain = np.array(xtrain), np.array(ytrain)
        xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1], 1))
        return xtrain, ytrain
    
    def figure_output(self, test_data, predict_data):
        plt.figure(figsize=(10, 5))
        plt.plot(test_data,'red', label = 'Real Prices')
        plt.plot(predict_data, 'blue', label = 'Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        plt.title('Real vs Predicted Prices')
        plt.legend(loc = 'best', fontsize = 20)
        plt.show()
        
    def model_predict(self):
        # testing data ready
        test_open = self.test_data.iloc[:, 0:1].values #taking  open price
        total = pd.concat([self.train_data['open'], self.test_data['open']], axis=0)
        locus = len(total) - len(self.test_data) - 1
        test_input = self.scaler.transform(total[locus:].values.reshape(-1,1))
        xtest = np.array([ test_input[i - 1 : i, 0] for i in range(1, 21) ]) #creating input for lstm prediction
        xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
        
        # predicting
        predicted_value = self.scaler.inverse_transform(self.model.predict(xtest))
        
        # evaluate
        print(mean_squared_error(test_open, predicted_value, squared = False))
        self.figure_output(test_data = test_open, predict_data = predicted_value)
        
        return [ value[0] for value in predicted_value.tolist() ]
    
    def making_action(self, msg):
        return {
            'BUY': 'HOLD' if self.stock == 1 else 'BUY',
            'HOLD': 'HOLD',
            'SELL': 'HOLD' if self.stock == -1 else 'SELL',
        }[msg]

    def get_action(self, action):
        return {
            'BUY': 1,
            'HOLD': 0,
            'SELL': -1
        }[action]
        
    def get_strategy(self, strategy, x_1, x_2):
        return {
            'portion': (x_2 - x_1) / (x_2 + x_1),
            'difference': (x_2 - x_1) / (x_2 + x_1),
        }[strategy]

    def main(self):
        # training data ready
        self.train_data, self.test_data = self.data_loader()
        xtrain, ytrain = self.data_preprocessing()
        
        # model ready
        self.model = self.model_building(input_size = (xtrain.shape[1], xtrain.shape[2]))
        
        # model training
        self.model.fit(xtrain, ytrain, batch_size=30, epochs=100)

        # prediction
        prediction = self.model_predict()
        
        # decision making 
        prediction_num = len(prediction)
        stock_num = 0
        opt_text = ''
        obs_day = 3
        for idx, price in enumerate(prediction):
            action = 0
            if idx + 1 <= (prediction_num - obs_day):
                trend = [ self.get_strategy('difference', Decimal(prediction[ idx + i + 1 ]), Decimal(price)) > 0 for i in range(obs_day) ]
                pos_portion = int(100 * trend.count(True) / (trend.count(True) + trend.count(False)))
                if pos_portion > 70:
                    action = self.get_action(self.making_action('BUY'))
                elif 70 >= pos_portion > 40:
                    action = self.get_action(self.making_action('HOLD'))
                else:
                    action = self.get_action(self.making_action('SELL'))
            else:
                if stock_num != 0:
                    action = self.get_action(self.making_action('SELL' if stock_num == 1 else 'BUY'))
                else:
                    action = self.get_action(self.making_action('HOLD'))
            self.stock += action
            if idx + 1 != prediction_num:
                opt_text += f'{action}\n'
            
        with open(self.output_filename, 'w') as f:
            f.writelines(opt_text)
            f.close()
        
        
if __name__ == '__main__':
    # You should not modify this part.
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    trader = Trader(train_filename = args.training, test_filename = args.testing, output_filename = args.output)
    trader.main()
    