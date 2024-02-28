import os
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

def find_time(folder_path):
    latest_start_time = datetime.min
    earliest_end_time = datetime.max
    # Create a DataFrame to store start and end times
    df_times = pd.DataFrame(columns=['start_time', 'end_time'])

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            if 'datetime' in df.columns:
                # Convert the 'timestamp' column to datetime
                df['datetime'] = pd.to_datetime(df['datetime'])

                # Find the latest start time and earliest end time in the current DataFrame
                current_start_time = df['datetime'].min()
                current_end_time = df['datetime'].max()

                # Concatenate start and end times to the DataFrame
                df_times = pd.concat(
                    [df_times, pd.DataFrame({'start_time': [current_start_time], 'end_time': [current_end_time]})],
                    ignore_index=True)

        print(filename)


            # Find the overall latest start time and earliest end time
    overall_latest_start_time = df_times['start_time'].max()
    overall_earliest_end_time = df_times['end_time'].min()

    return overall_latest_start_time, overall_earliest_end_time


def process_csv_files(folder_path, start_time, end_time, db):

    # Create a DataFrame to store hourly aggregations
    df_hourly_aggregations = pd.DataFrame(columns=['datetime', 'hourly_average_price', 'sum_liquidity'])

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            if 'datetime' in df.columns:
                # Convert the 'datetime' column to datetime
                df['datetime'] = pd.to_datetime(df['datetime'])

                # Filter data based on given start time and end time
                df_filtered = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]

                # Group by hourly intervals and calculate mean price and sum of liquidity
                df_hourly = df_filtered.groupby(pd.Grouper(key='datetime', freq='1H')).agg({
                    'vw': 'mean',
                    'n': 'sum'
                }).reset_index()

                # Append hourly aggregations to the DataFrame
                # Store df_hourly_aggregations in MongoDB
                collection_name = os.path.splitext(filename)[0] + '_hourly_aggregations'
                db[collection_name].insert_many(df_hourly.to_dict(orient='records'))


# Example usage
if __name__ == "__main__":
    # Specify the directory containing CSV files
    folder_path = '/Users/hengyi_wang/Downloads/drive-download-20231111T221652Z-001/historical_fx_rates'

    #latest_start, earliest_end = find_time(folder_path)
    # Latest Start Time: 2021-01-08 00:00:00
    # Earliest End Time: 2023-01-24 23:54:00

    # Connect to MongoDB
    client = MongoClient('localhost', 27017)
    database = client["HW3_fx_database"]

    latest_start = datetime(2021, 1, 8, 0, 0, 0)
    earliest_end = datetime(2023, 1, 24, 23, 54, 0)

    process_csv_files(folder_path, latest_start, earliest_end, database)

