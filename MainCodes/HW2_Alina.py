import sqlite3
import pandas as pd
import numpy as np
import pymongo
import datetime
import time
from polygon import RESTClient
import os
from pymongo import MongoClient



# Step 1: Fetch FX rates from Polygon API
def get_fx_rates(ticker):
    client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")
    from_curr = ticker[0:3]
    to_curr = ticker[3:]
    result = client.get_real_time_currency_conversion(from_=from_curr, to=to_curr, )
    timestamp = datetime.datetime.fromtimestamp(result.last.timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

    return ticker[:], timestamp, result.converted

# Function to retrieve raw data from MongoDB
def retrieve_raw_data(mongo_db):
    # Example: Assuming you have a MongoDB collection named 'raw_data'
    raw_data_collection = mongo_db.raw_data

    # Retrieve all documents from the 'raw_data' collection
    raw_data = list(raw_data_collection.find())
    print(raw_data)
    return raw_data


# Function to calculate the Fractal Dimension (FD)
def calculate_fractal_dimension(max_value, min_value, fx_rate_values):
    # Calculate the number of times the price crosses the Keltner Channel
    crosses_channel = np.sum((fx_rate_values > max_value) | (fx_rate_values < min_value))

    # Calculate FD (Fractal Dimension)
    max_min_diff = max_value - min_value
    fd = crosses_channel / max_min_diff if max_min_diff != 0 else 0.0

    return fd


# Step 4: Calculate auxiliary data for a specific 6-minute interval, including FD
def calculate_auxiliary_data(conn):
    # loads data as pandas DataFrame
    df = pd.read_sql_query("select * from fx_rates;", conn)

    # Convert raw data to a DataFrame
    df = pd.DataFrame(df)

    # df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    # # Organize the data
    # df['timestamp_rounded'] = df['entry_timestamp'].dt.floor('6T')  # Round timestamps to the nearest 6 minutes

    # Group the data by 6-minute intervals
    # dict key:val
    grouped_data = df.groupby(by='fx_timestamp')

    # Calculate auxiliary data for each group, including FD
    auxiliary_data = []
    for groups in grouped_data:
        currency_pair = groups
        # 3: fx_rate
        fx_timestamp = groups[1]['entry_timestamp'].iloc[0]
        max_value = groups[1]['fx_rate'].max()
        min_value = groups[1]['fx_rate'].min()
        mean_value = groups[1]['fx_rate'].mean()
        vol = groups[1]['fx_rate'].std()  # Assuming standard deviation as a measure of volatility

        # Calculate FD for the group
        fd = calculate_fractal_dimension(max_value, min_value, groups[1]['fx_rate'])

        auxiliary_data.append({
            'fx_timestamp': fx_timestamp,
            'currency_pair': currency_pair[0],
            'max_value': max_value,
            'min_value': min_value,
            'mean_value': mean_value,
            'vol': vol,
            'fd': fd
        })

    # starttime = time.monotonic()
    # starttime = str(starttime)
    # df = pd.DataFrame(auxiliary_data,columns=['fx_timestamp', 'currency_pair', 'max_value','min_value','mean_value','vol','fd'])
    # df.to_csv("auxiliary_data"+starttime+".csv",index=False)
    # Create a temporary SQLite database for the 6-minute interval
    temp_db_name = 'temp_auxiliary_data_with_fd.db'
    temp_conn = sqlite3.connect(temp_db_name)
    temp_cursor = temp_conn.cursor()

    # Create a table for auxiliary data
    temp_cursor.execute('CREATE TABLE IF NOT EXISTS auxiliary_data_with_fd ('
                        'fx_timestamp DATETIME, '
                        'currency_pair TEXT, '
                        'max_value REAL, '
                        'min_value REAL, '
                        'mean_value REAL, '
                        'vol REAL, '
                        'fd REAL)')

    # starttime = time.monotonic()
    # starttime = str(starttime)
    # df = pd.DataFrame(auxiliary_data,columns=['fx_timestamp', 'currency_pair', 'max_value','min_value','mean_value','vol','fd'])
    # df.to_csv("auxiliary_data"+starttime+".csv",index=False)

    # Insert calculated results into the temporary SQLite database
    for entry in auxiliary_data:
        temp_cursor.execute('INSERT INTO auxiliary_data_with_fd VALUES (?, ?, ?, ?, ?, ?, ?)',
                            (entry['fx_timestamp'], entry['currency_pair'],
                             entry['max_value'], entry['min_value'],
                             entry['mean_value'], entry['vol'], entry['fd']))

    # Commit the changes to the temporary database
    temp_conn.commit()
    temp_conn.close()

    return temp_db_name


# Step 5: Store results in the Final DB
def transfer_to_final_db(temp_db_name, mongo_db):
    # Connect to the temporary SQLite database
    temp_conn = sqlite3.connect(temp_db_name)
    temp_cursor = temp_conn.cursor()

    # Query the temporary SQLite database for calculated results
    temp_cursor.execute('SELECT * FROM auxiliary_data_with_fd')
    auxiliary_data = temp_cursor.fetchall()

    # Create or get a reference to the MongoDB collection
    final_db_collection = mongo_db.final_data

    # Insert calculated results into the MongoDB collection
    for entry in auxiliary_data:
        final_db_collection.insert_one({
            'fx_timestamp': entry[0],
            'currency_pair': entry[1],
            'max_value': entry[2],
            'min_value': entry[3],
            'mean_value': entry[4],
            'vol': entry[5],
            'fd': entry[6]
        })

    # Close the connections and delete the temporary SQLite database
    temp_conn.close()
    temp_db_deleted = False
    try:
        os.remove(temp_db_name)
        temp_db_deleted = True
    except Exception as e:
        print(f"Error deleting temporary database: {e}")

    if temp_db_deleted:
        print(f'Temporary database {temp_db_name} deleted.')
    print(f'data inserted into MongoDB collection')

def create_table(conn):
    try:
        cursor = conn.cursor()

        # Replace 'your_table' with the desired table name
        table_name = 'fx_table'

        # Replace 'column1', 'column2', etc. with your column names and data types
        # For example, 'id INTEGER PRIMARY KEY', 'name TEXT', 'age INTEGER'
        # define SQL query to create table
        create_table_query = """ CREATE TABLE IF NOT EXISTS fx_rates (
                                                id integer PRIMARY KEY,
                                                fx_timestamp text NOT NULL,
                                                ticker text NOT NULL,
                                                fx_rate real NOT NULL,
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

    sql = ''' INSERT INTO fx_rates(fx_timestamp, ticker, fx_rate, entry_timestamp)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, fx)
    conn.commit()

    return cur.lastrowid

def main():
    # Step 1
    # Step 1: Set up connections
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
    mongo_db = mongo_client["fx_fxdatabase_v1"]
    # define DB file
    sqlite_file = './fx_rates.db'
    # connecting to the database file
    conn = sqlite3.connect(sqlite_file)
    # raw_fx_data = []
    # create table
    create_table(conn)
    # Step 2
    tickers = ["EURUSD", "GBPEUR", "USDAUD"]
    starttime = datetime.datetime.now()
    starttime_1 = time.monotonic()
    deltatime = starttime
    # Run the data collection loop for 5 hours
    while abs(pd.Timestamp.now() - starttime) <= pd.Timedelta(5, unit="hours"):
        for ticker in tickers:
            entry = get_fx_rates(ticker) + ((str(datetime.datetime.now()).split('.')[0]),)
            create_entry(conn, entry)
        if abs(pd.Timestamp.now() - deltatime) >= pd.Timedelta(6, unit="minutes"):
            # Calculate auxiliary data and get the temporary database name
            temp_db_name = calculate_auxiliary_data(conn)
            # Transfer data to the final MongoDB database
            transfer_to_final_db(temp_db_name, mongo_db)
            deltatime = deltatime + pd.Timedelta(6, unit="minutes")
        time.sleep(1.0 - ((time.monotonic() - starttime_1) % 1.0))
        # raw_fx_data.append(raw_data)
        # for every 6 mins data
    # close the connection
    conn.close()
    mongo_client.close()


# Execute the main function
if __name__ == "__main__":
    main()