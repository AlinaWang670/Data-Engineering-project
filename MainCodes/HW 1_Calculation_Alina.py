import pandas as pd
from pymongo import MongoClient


# Function to retrieve raw data from MongoDB
def retrieve_raw_data(collection):
    # Retrieve a collection named "user_1_items" from database
    collection_name = collection
    item_details = collection_name.find()
    # convert the dictionary objects to dataframe
    items_df = pd.DataFrame(item_details)
    return items_df



# Example usage
def main():
    #connection
    # Create a connection using MongoClient.
    client = MongoClient('localhost', 27017)
    # Create the database for our example
    dbname = client['fx_mgdatabase']
    collection = dbname["fx_collection"]
    raw_df = retrieve_raw_data(collection)
    #print(raw_df)
    avg_fx_df = pd.DataFrame(raw_df.loc[:,['currency_pair','fx_rate']])
    #print(avg_fx_df)
    result = avg_fx_df.groupby(['currency_pair']).mean()
    print(result)
    client.close()


# Execute the main function
if __name__ == "__main__":
    main()