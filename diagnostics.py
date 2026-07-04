import json
import os
import pickle
import subprocess
import sys
import timeit

import pandas as pd


################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

FEATURE_COLUMNS = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET_COLUMN = 'exited'


################## Function to get model predictions
def model_predictions(test_data):
    # Read the deployed model and a dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(test_data[FEATURE_COLUMNS])
    return predictions.tolist()


################## Function to get summary statistics
def dataframe_summary():
    # Calculate summary statistics here
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    numeric_data = data.select_dtypes(include=['number'])

    means = numeric_data.mean().tolist()
    medians = numeric_data.median().tolist()
    stds = numeric_data.std().tolist()

    return [means, medians, stds]


################## Function to get percent of missing data
def missing_data():
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    return (data.isna().mean() * 100).tolist()


##################Function to get timings
def execution_time():
    # Calculate timing of ingestion.py and training.py
    start_time = timeit.default_timer()
    subprocess.run([sys.executable, 'ingestion.py'], check=True)
    ingestion_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    subprocess.run([sys.executable, 'training.py'], check=True)
    training_time = timeit.default_timer() - start_time

    return [ingestion_time, training_time]


################## Function to check dependencies
def outdated_packages_list():
    # Get a list of current and latest package versions
    requirements = {}
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '==' not in line:
                continue
            package, version = line.split('==', 1)
            requirements[package] = version

    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
        check=True,
        capture_output=True,
        text=True,
    )
    outdated_packages = {
        item['name']: item['latest_version']
        for item in json.loads(result.stdout or '[]')
    }

    rows = []
    for package, current_version in requirements.items():
        rows.append({
            'package': package,
            'current_version': current_version,
            'latest_version': outdated_packages.get(package, current_version),
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    print(model_predictions(data))
    print(dataframe_summary())
    print(missing_data())
    print(execution_time())
    print(outdated_packages_list())
