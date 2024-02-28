from fredapi import Fred
import pandas as pd
import numpy as np
from scipy.interpolate import splev, splrep
from pymongo import MongoClient

fred = Fred(api_key='693192912bac0154eb545bdb0d5ad49d')

indicators = {
    'GDP': 'GDPC1',
    'Unemployment': 'UNRATE',
    'CPI': 'CPIAUCNS',
    'Core CPI': 'CPILFESL'
}

start_date = '2013-10-01'
end_date = '2023-10-01'

economic_data = pd.DataFrame()

for indicator_name, indicator_code in indicators.items():
    data = fred.get_series(indicator_code, observation_start=start_date, observation_end=end_date)
    economic_data[indicator_name] = data

daily_data = economic_data.resample('D').asfreq()

daily_data = daily_data.ffill()

start_date = economic_data.index[0]

# Upscale the original time series with a spline
def upscale_with_spline(data):
    x = np.arange(len(data))
    y = data.values
    spline = splrep(x, y, k=3)
    x_new = np.linspace(0, len(data) - 1, len(data) * 5)
    y_new = splev(x_new, spline)
    new_index = pd.date_range(start=start_date, periods=len(data) * 5, freq='D')
    return pd.Series(y_new, index=new_index)

upscaled_data = daily_data.apply(upscale_with_spline)

# Exclude weekends (Saturday and Sunday)
upscaled_data = upscaled_data[upscaled_data.index.dayofweek < 5]

print("Upscaled Data:")
print(upscaled_data)

upscaled_df = pd.DataFrame(upscaled_data)

upscaled_df.to_csv('upscaled_data.csv', index=True)

# Store data in MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['final_database']
collection = db['upscaled_data']

upscaled_dict = upscaled_df.reset_index().to_dict(orient='records')

collection.insert_many(upscaled_dict)

client.close()