import json
import os

import pandas as pd


############# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


############# Function for data ingestion
def merge_multiple_dataframe():
    # Check for datasets, compile them together, and write to an output file
    csv_files = sorted(
        file_name
        for file_name in os.listdir(input_folder_path)
        if file_name.endswith('.csv')
    )

    dataframes = [
        pd.read_csv(os.path.join(input_folder_path, file_name))
        for file_name in csv_files
    ]

    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True).drop_duplicates()
    else:
        final_df = pd.DataFrame()

    os.makedirs(output_folder_path, exist_ok=True)

    final_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write(str(csv_files))

    return final_df


if __name__ == '__main__':
    merge_multiple_dataframe()
