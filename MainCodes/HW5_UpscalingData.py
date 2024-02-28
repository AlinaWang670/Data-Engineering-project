import pandas as pd
import numpy as np
from scipy.interpolate import splev, splrep

economic_data = pd.read_csv('economic_data.csv', index_col=0, parse_dates=True)

print("Original Data:")
print(economic_data)

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