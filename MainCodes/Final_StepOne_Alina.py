'''
Collecting a 10-hour fx data for the targeted currency pairs for regression
'''
import sqlite3
import pandas as pd
import pymongo
import datetime
import time
from polygon import RESTClient

# Fetch FX rates from Polygon API
def get_fx_rates(ticker):
    client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")
    from_curr = ticker[0:3]
    to_curr = ticker[3:]
    result = client.get_real_time_currency_conversion(from_=from_curr, to=to_curr, )
    timestamp = datetime.datetime.fromtimestamp(result.last.timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    forex_rate = round((result.last.ask + result.last.bid) / 2, 6)

    return ticker[:], timestamp, forex_rate, result.last.exchange

def calculate_features(conn):
    # loads data as pandas DataFrame
    df = pd.read_sql_query("select * from fx_rates;", conn)
    df = pd.DataFrame(df)
    # Convert 'fx_timestamp' column to datetime object
    df['fx_timestamp'] = pd.to_datetime(df['fx_timestamp'])
    # Get unique tickers
    unique_tickers = df['ticker'].unique()
    data_vectors = []  # Initialize an empty list to store data vectors

    for ticker in unique_tickers:
        # Filter DataFrame for the current ticker
        ticker_df = df[df['ticker'] == ticker]
        # Find the minimum and maximum datetime values
        min_datetime = ticker_df['fx_timestamp'].min()
        max_datetime = ticker_df['fx_timestamp'].max()
        # Set 'datetime' column as index
        ticker_df.set_index('fx_timestamp', inplace=True)
        ticker_df.rename(columns={'vw': 'price'}, inplace=True)

        # Resample the data to 6-minute intervals
        resampled_df = ticker_df.resample('6T').agg({'price': 'ohlc', 'n': 'sum'})
        # Calculate VWAP
        resampled_df['vwap'] = (resampled_df['price']['close'] * resampled_df['n']['n']).div(
            resampled_df['n']['n'].sum())
        # Calculate mean
        resampled_df['mean'] = (resampled_df['price']['open'] + resampled_df['price']['high'] +
                                resampled_df['price']['low'] + resampled_df['price']['close']) / 4
        # Calculate additional features
        resampled_df['max_min_diff'] = resampled_df['price', 'high'] - resampled_df['price', 'low']
        resampled_df['vol'] = resampled_df['max_min_diff'] / resampled_df['mean']
        # Resample the data to 6-minute intervals
        resampled_df_1 = ticker_df.resample('6T').agg({'n': 'sum'})
        # Calculate Liquidity
        resampled_df_1['liquidity'] = resampled_df_1['n'].mean()

        keltner_upper_bands = []
        keltner_lower_bands = []

        for i in range(1, 100 + 1):
            keltner_upper_band = resampled_df.iloc[0]['mean'] + i * 0.025 * resampled_df.iloc[0][
                'vol']
            keltner_lower_band = resampled_df.iloc[0]['mean'] - i * 0.025 * resampled_df.iloc[0][
                'vol']

            keltner_upper_bands.append(keltner_upper_band)
            keltner_lower_bands.append(keltner_lower_band)

        # Convert the lists to Pandas Series
        keltner_upper_series = pd.Series(keltner_upper_bands, name='keltner_upper_bands')
        keltner_lower_series = pd.Series(keltner_lower_bands, name='keltner_lower_bands')

        # Combine the upper and lower bands into a DataFrame
        keltner_bands = pd.concat([keltner_upper_series, keltner_lower_series], axis=1)

        # Reset index to have 'datetime' as a regular column
        resampled_df.reset_index(inplace=True)

        # Get the last timestamp in the resampled data
        last_timestamp = resampled_df.iloc[0]['fx_timestamp']

        # Create the current_time with hours, minutes, and seconds information
        current_time = last_timestamp + pd.Timedelta(minutes=6)

        # Convert last_timestamp to timestamp with detailed time information
        # current_time = pd.to_datetime(current_time)
        ct = current_time[0]

        current_period_data = ticker_df[ticker_df.index > ct]
        ct = ct + pd.Timedelta(minutes=6)
        current_period_data = current_period_data[current_period_data.index < ct]

        # Count the number of times the price crosses the Keltner Channel
        crosses_keltner = ((current_period_data['price'] > keltner_bands.iloc[0, 0][0]) |
                           (current_period_data['price'] < keltner_bands.iloc[0, 1][0])).sum()
        if len(resampled_df) > 1:
            fd = crosses_keltner / resampled_df.iloc[1]['max_min_diff']
            fd = float(fd)
            # Build the data vector for the current ticker
            data_vector = {
                'ticker': ticker,
                'timestamp': ct,
                'vwap': resampled_df['vwap'][1],
                'liq': resampled_df_1['liquidity'].iloc[1],
                'vol': resampled_df['vol'][1],
                'max': resampled_df['price', 'high'][1],
                'min': resampled_df['price', 'low'][1],
                'fd': fd
            }

            data_vectors.append(data_vector)

    return data_vectors


def transfer_to_final_db(data_vectors, mongo_db):
    collection = mongo_db['fx_data']

    try:
        for data_vector in data_vectors:
            ticker = data_vector['ticker']
            existing_document = collection.find_one({'ticker': ticker})
            if existing_document:
                collection.update_one({'ticker': ticker}, {'$push': {'data_vectors': data_vector}})
            else:
                new_document = {
                    'ticker': ticker,
                    'data_vectors': [data_vector]
                }
                collection.insert_one(new_document)
        print("Data vectors stored in MongoDB.")
    except Exception as e:
        print(f"Error storing data vectors in MongoDB: {e}")

def create_table(conn):
    try:
        cursor = conn.cursor()
        table_name = 'fx_table'
        # define SQL query to create table
        create_table_query = """ CREATE TABLE IF NOT EXISTS fx_rates (
                                                id integer PRIMARY KEY,
                                                fx_timestamp text NOT NULL,
                                                ticker text NOT NULL,
                                                vw REAL,
                                                n REAL,
                                                entry_timestamp text NOT NULL
                                            ); """
        cursor.execute(create_table_query)
        conn.commit()

        print(f"Table {table_name} created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")


def create_entry(conn, fx):
    """
    Insert a new fx rate entry into the DB
    :param conn: Connection object
    :param fx: row data obtained from Polygon.io
    :return:
    """
    sql = ''' INSERT INTO fx_rates(ticker, fx_timestamp, vw, n, entry_timestamp)
              VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, fx)
    conn.commit()

    return cur.lastrowid

def main():
    # Step 1: Set up connections
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
    mongo_db = mongo_client["final_database"]
    # Step 2
    tickers = ["USDEUR", "USDGBP", "USDAUD","USDCNY","USDHKD","USDJPY","USDPLN"]
    starttime = datetime.datetime.now()
    starttime_1 = time.monotonic()
    deltatime = starttime
    # define DB file
    sqlite_file = './final_fx_rates.db'
    # connecting to the database file
    conn = sqlite3.connect(sqlite_file)
    # create table
    create_table(conn)
    # Run the data collection loop for 10 hours
    while abs(pd.Timestamp.now() - starttime) <= pd.Timedelta(10, unit="hours"):
        for ticker in tickers:
            entry = get_fx_rates(ticker) + ((str(datetime.datetime.now()).split('.')[0]),)
            create_entry(conn, entry)
        if abs(pd.Timestamp.now() - deltatime) >= pd.Timedelta(13, unit="minutes"):
            # Calculate auxiliary data and get the temporary database name
            data_vectors = calculate_features(conn)
            # Transfer data to the final MongoDB database
            transfer_to_final_db(data_vectors, mongo_db)
            deltatime = deltatime + pd.Timedelta(6, unit="minutes")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM fx_rates WHERE fx_timestamp < ?", (deltatime,))
            conn.commit()
            print(f'Delete Old Temporary data.')
            print(f'data inserted into MongoDB collection')
        time.sleep(1.0 - ((time.monotonic() - starttime_1) % 1.0))
    mongo_client.close()


# Execute the main function
if __name__ == "__main__":
    main()