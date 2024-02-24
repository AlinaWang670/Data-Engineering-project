import pymongo
import pandas as pd
import numpy as np
from pycaret.regression import *


class TradingStrategy:
    def __init__(self, ticker, exogenous_data):
        self.ticker = ticker
        self.data = self.load_data_from_mongodb()
        self.regression_model = self.train_regression_model()
        self.extended_data = pd.concat([self.data, exogenous_data], axis=1)
        self.extended_data = self.extended_data.dropna(subset=['timestamp'])
        self.correlation_matrix = self.calculate_correlation()

    def calculate_correlation(self):
        correlation_matrix = self.extended_data[
            ['vwap', 'liq', 'vol', 'fd', 'max', 'min', 'GDP', 'Unemployment', 'CPI', 'Core CPI']].corr()
        # print(correlation_matrix)
        return correlation_matrix

    def load_data_from_mongodb(self):
        # Connect to MongoDB and fetch data for the specific ticker
        client = pymongo.MongoClient("mongodb://localhost:27017")
        db = client["final_database"]
        collection = db[f"fx_data_{self.ticker}"]

        # Retrieve all documents from the collection
        data_documents = list(collection.find())

        # Create DataFrame directly from MongoDB data
        data = pd.DataFrame(data_documents)

        return data

    def train_regression_model(self, target_column='vwap'):
        print(f"Training new model for {self.ticker}")
        dataset = self.data

        # Replace infinity values with NaN and then drop rows with NaN
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(inplace=True)

        # Initialize setup with the specific data
        reg_setup = setup(self.data, target=target_column, session_id=123, log_experiment=True,
                          experiment_name=f'{self.ticker}_regression')

        # Compare models and select the best one
        best_model = compare_models()

        # Save the trained model with a unique name based on the ticker
        save_model(best_model, model_name=f"{self.ticker}_reg_model")

        return best_model

    def generate_signals(self):
        # use the regression model to predict future values
        predictions = predict_model(self.regression_model, data=self.data)

        self.data['predicted_vwap'] = predictions['prediction_label']

        #Trading Strategy
        self.data['signal'] = np.where(
            np.logical_or(
                self.data['predicted_vwap'] < self.data['vwap'],
                np.logical_or(
                    self.correlation_matrix.loc['Unemployment', 'vwap'] > 0.3,
                    self.correlation_matrix.loc['CPI', 'vwap'] > 0.3
                )
            ),
            'buy', 'sell'
        )

        return self.data[['timestamp', 'vwap', 'predicted_vwap', 'signal']]


client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["final_database"]
collection = db["upscaled_data"]
data_documents = list(collection.find())
exogenous_data = pd.DataFrame(data_documents)

for ticker in ['USDEUR', 'USDGBP', 'USDAUD', 'USDCNY', 'USDHKD', 'USDJPY', 'USDPLN']:
    strategy = TradingStrategy(ticker, exogenous_data)
    signals = strategy.generate_signals()
    # Save the signals DataFrame to a CSV file
    signals.to_csv(f"{ticker}_signals.csv", index=False)
    print(signals)
