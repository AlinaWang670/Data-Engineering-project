from pymongo import MongoClient
import pandas as pd

def calculate_fd_features(collection):
    # Query documents within the specified time range
    query = {}
    df = pd.DataFrame(list(collection.find(query)))

    # Convert 'datetime' column to datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Find the minimum and maximum datetime values
    min_datetime = df['datetime'].min()
    max_datetime = df['datetime'].max()

    # Calculate the total number of six-hour intervals
    six_hour_intervals = pd.date_range(start=min_datetime, end=max_datetime, freq='6H')

    # Set 'datetime' column as index
    df.set_index('datetime', inplace=True)
    df.rename(columns={'vw': 'price'}, inplace=True)

    # Initialize variables to store FD and other features
    data_vectors = []

    # Iterate through 6-hour buckets within the second to the 50th period
    for period in range(2, len(six_hour_intervals)):
        # Resample the data to 6-hour intervals
        resampled_df = df.resample('6H').agg({'price': 'ohlc', 'n': 'sum'})
        # Calculate VWAP
        resampled_df['vwap'] = (resampled_df['price']['close'] * resampled_df['n']['n']).div(
            resampled_df['n']['n'].sum())
        # Calculate mean
        resampled_df['mean'] = (resampled_df['price']['open'] + resampled_df['price']['high'] + resampled_df['price'][
            'low'] + resampled_df['price']['close']) / 4
        # Calculate additional features
        resampled_df['max_min_diff'] = resampled_df['price', 'high'] - resampled_df['price', 'low']
        resampled_df['vol'] = resampled_df['max_min_diff'] / resampled_df['mean']
        # Resample the data to 6-hour intervals
        resampled_df_1 = df.resample('6H').agg({'n': 'sum'})
        # Calculate Liquidity (average number of transactions per hour)
        resampled_df_1['liquidity'] = resampled_df_1['n']/6

        keltner_upper_bands = []
        keltner_lower_bands = []

        for i in range(1, 100 + 1):
            keltner_upper_band = resampled_df.iloc[period-2]['mean'] + i * 0.025 * resampled_df.iloc[period-2]['vol']
            keltner_lower_band = resampled_df.iloc[period-2]['mean'] - i * 0.025 * resampled_df.iloc[period-2]['vol']

            keltner_upper_bands.append(keltner_upper_band[0])
            keltner_lower_bands.append(keltner_lower_band[0])

        # Convert the list to a Pandas DataFrame
        keltner_bands = pd.DataFrame({'keltner_upper_bands': keltner_upper_bands,
                           'keltner_lower_bands': keltner_lower_bands})

        # Reset index to have 'datetime' as a regular column
        resampled_df.reset_index(inplace=True)
        # Get the last timestamp in the resampled data
        last_timestamp = resampled_df.iloc[0]['datetime']
        # Create the current_time with hours, minutes, and seconds information
        current_time = last_timestamp + pd.Timedelta(hours=(period - 1) * 6)
        # Convert last_timestamp to timestamp with detailed time information
        current_time = pd.to_datetime(current_time)
        ct = current_time.iloc[0]

        # current_period_data = df.index[df.index > ct]
        current_period_data = df[df.index > ct]

        ct = ct + pd.Timedelta(hours=6)
        current_period_data = current_period_data[current_period_data.index < ct]

        # Count the number of times the price crosses the Keltner Channel
        crosses_keltner = ((current_period_data['price'] > keltner_bands.iloc[period-2,0]) |
                           (current_period_data['price'] < keltner_bands.iloc[period-2,1])).sum()

        # Calculate FD
        fd = crosses_keltner / resampled_df.iloc[period-1]['max_min_diff']

        # Build the data vector
        data_vector = {
            'timestamp': ct,
            'vwap':resampled_df['vwap'][period-1],
            'liq':resampled_df_1['liquidity'].iloc[period-1],
            'vol': resampled_df['vol'][period-1],
            'max': resampled_df['price', 'high'][period-1],
            'min': resampled_df['price', 'low'][period-1],
            'fd': fd.iloc[0]
        }

        # Append the data vector to the list
        data_vectors.append(data_vector)

    return data_vectors

# Connect to MongoDB
client = MongoClient('localhost', 27017)
database = client["HW3_fx_database"]
# Iterate through each collection
for collection_name in database.list_collection_names():
        collection = database[collection_name]
        data_vectors = calculate_fd_features(collection)
        # Convert the list of dictionaries to a DataFrame
        df_data_vectors = pd.DataFrame(data_vectors)
        # Convert the list to a DataFrame
        df = pd.DataFrame(df_data_vectors, columns=['timestamp', 'vwap', 'liq', 'vol', 'max', 'min', 'fd'])
        # Write the DataFrame to the CSV file
        #print(df)
        df.to_csv(f'{collection_name} df_data_vectors.csv', index=False)
        print(f'{collection_name} has been processed')
