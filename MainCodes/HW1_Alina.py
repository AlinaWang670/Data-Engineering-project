#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:13:47 2023

@author: hengyi_wang
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:10:58 2023

@author: hengyi_wang
"""
import pandas as pd
import datetime
import time
from polygon import RESTClient
from pymongo import MongoClient



# Step 2: Fetch FX rates from Polygon API
def get_fx_rates(ticker):
    client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")
    from_curr = ticker[0:3]
    to_curr = ticker[3:]
    result = client.get_real_time_currency_conversion(from_=from_curr, to=to_curr, )
    timestamp = datetime.datetime.fromtimestamp(result.last.timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

    return ticker[:], timestamp, result.converted


# Step 3: Store data in MongoDB
def store_raw_data(collection, entry):
    # Insert all documents from raw_data list into the MongoDB collection
    # Create or get a reference to the MongoDB collection
    collection.insert_one({
        'currency_pair': entry[0],
        'fx_timestamp': entry[1],
        'fx_rate': entry[2],
        'insert_time': entry[3]
    })
    #print(f'Raw data inserted into MongoDB collection: raw_data')


def main():
    #connection
    # Create a connection using MongoClient.
    client = MongoClient('localhost', 27017)
    # Create the database for our example
    dbname = client['fx_mgdatabase']
    collection = dbname["fx_collection"]
    # # Step 2
    tickers = ["EURUSD", "GBPEUR", "USDAUD"]
    starttime = datetime.datetime.now()
    starttime_1 = time.monotonic()
    deltatime = starttime
    # Run the data collection loop for 5 hours
    while abs(pd.Timestamp.now() - starttime) <= pd.Timedelta(3, unit="hours"):
        for ticker in tickers:
            entry = get_fx_rates(ticker) + ((str(datetime.datetime.now()).split('.')[0]),)
            store_raw_data(collection,entry)
        time.sleep(1.0 - ((time.monotonic() - starttime_1) % 1.0))
    client.close()


# Execute the main function
if __name__ == "__main__":
    main()