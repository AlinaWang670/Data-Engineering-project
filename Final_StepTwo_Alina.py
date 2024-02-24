import pymongo
import pandas as pd

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["final_database"]
collection = db["fx_data"]

# Retrieve all documents from the collection
data_documents = list(collection.find())

# Reorganize data by ticker
reorganized_data = {}

for document in data_documents:
    ticker = document['ticker']
    data_vectors = document['data_vectors']

    if ticker not in reorganized_data:
        reorganized_data[ticker] = []

    for entry in data_vectors:
        reorganized_entry = {
            'timestamp': entry['timestamp'],
            'vwap': entry['vwap'],
            'liq': entry['liq'],
            'vol': entry['vol'],
            'max': entry['max'],
            'min': entry['min'],
            'fd': entry['fd'],
            # Add other fields as needed
        }

        reorganized_data[ticker].append(reorganized_entry)

# Create DataFrames for each ticker
ticker_dataframes = {}

for ticker, entries in reorganized_data.items():
    ticker_dataframes[ticker] = pd.DataFrame(entries)

    # Create a new collection for each ticker
    ticker_collection = db[f"fx_data_{ticker}"]

    # Convert DataFrame to dictionary and insert into MongoDB collection
    records = entries
    ticker_collection.insert_many(records)

    print(f"Data for {ticker} stored in collection {ticker_collection.name}")
