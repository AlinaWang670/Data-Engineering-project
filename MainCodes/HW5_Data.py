from fredapi import Fred
import pandas as pd

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

economic_data.to_csv('economic_data.csv', index=True)