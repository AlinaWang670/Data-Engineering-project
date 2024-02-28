import pandas as pd
import os

folder_path = '/Users/hengyi_wang/Desktop/NYU/Data Engineering/HW_Alina/simulator_v2 copy'

result_dict = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        df = pd.read_csv(file_path)

        df['PnL'] = 0

        # Correct the calculation of 'PnL' based on 'Direction'
        df.loc[df['Direction'] == 1, 'PnL'] = df['Exe_Price'] * df['Exe_Units']
        df.loc[df['Direction'] == -1, 'PnL'] = -df['Exe_Price'] * df['Exe_Units']

        print(df[['Direction', 'Exe_Price', 'Exe_Units', 'PnL']])
        Process_df = df.groupby('Direction')['PnL'].sum().reset_index()
        Total_PnL = Process_df.iloc[1, 1] - Process_df.iloc[0, 1]

        result_dict.append({'Total_PnL': Total_PnL, 'Fx_name': df['Asset_Name'][0]})

# Create a DataFrame from the dictionary
result_df = pd.DataFrame(result_dict)

print(result_df)
result_df.to_csv('Profit_and_Loss.csv')