import pandas as pd
from polygon import RESTClient

# Step 2: Fetch FX rates from Polygon API
def get_fx_rates(ticker):
    client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")
    from_curr = 'USD'
    to_curr = ticker
    result = client.get_real_time_currency_conversion(from_=from_curr, to=to_curr, )

    return result.converted


def main():
    df = pd.read_csv('Trading_Schedule.csv')
    FX_Names_Direction = df[['FX_Name','Direction']]
    ConverRate = []
    for index, row in FX_Names_Direction.iterrows():
        if row['Direction'] == 1:
            ticker = row['FX_Name'][-3:]
            if ticker == 'USD':
                ConverRate.append(100)
            else:
                ConverRate.append(get_fx_rates(ticker)*100)
        else:
            ticker = row['FX_Name'][:3]
            if ticker == 'USD':
                ConverRate.append(100)
            else:
                ConverRate.append(get_fx_rates(ticker)*100)
    df['Units'] = ConverRate
    # Round the 'Units' column to two decimal places
    df['Units'] = df['Units'].round(decimals=2)
    df.to_csv('Trading_Schedule_Converted.csv', index=False)


if __name__ == "__main__":
    main()