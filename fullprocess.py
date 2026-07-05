"""Process automation for the ML model scoring and monitoring pipeline.

Re-deployment decision process:
1. Check for new data. If none exists, stop; otherwise ingest it.
2. Check for model drift by scoring the currently deployed model on the
   newly ingested data. If performance has not degraded, stop.
3. Re-train a candidate model on the new data and score it on the test set.
4. Deploy the candidate model only if it outperforms the currently
   deployed model. A model that scores worse is never deployed.
5. After a successful re-deployment, regenerate the confusion matrix and
   API reports for the newly deployed model (saved as confusionmatrix2.png
   and apireturns2.txt, distinct outputs of this second run).
"""
import ast
import json
import os
import pickle
import subprocess
import sys
import time

import pandas as pd
from sklearn.metrics import f1_score

import deployment
import reporting


with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']

FEATURE_COLUMNS = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET_COLUMN = 'exited'


################## Check and read new data
def read_ingested_files():
    """Read the record of previously ingested files from the deployment directory."""
    ingested_file_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    if not os.path.exists(ingested_file_path):
        return []

    with open(ingested_file_path, 'r') as f:
        content = f.read().strip()

    return ast.literal_eval(content) if content else []


def discover_source_files():
    """List every csv file currently present in the input folder."""
    return sorted(
        file_name
        for file_name in os.listdir(input_folder_path)
        if file_name.endswith('.csv')
    )


################## Checking for model drift
def score_deployed_model_on_new_data():
    """Score the currently deployed model against the newly ingested data."""
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        deployed_model = pickle.load(f)

    new_data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    predictions = deployed_model.predict(new_data[FEATURE_COLUMNS])

    return f1_score(new_data[TARGET_COLUMN], predictions)


################## Diagnostics and reporting on the newly deployed model
def start_api_server():
    server = subprocess.Popen(
        [sys.executable, 'app.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return server


def run_reporting_and_api_calls():
    """Generate the second-run reports for the newly deployed model."""
    reporting.score_model('confusionmatrix2.png')

    server = start_api_server()
    try:
        subprocess.run([sys.executable, 'apicalls.py', 'apireturns2.txt'], check=True)
    finally:
        server.terminate()
        server.wait(timeout=10)


def main():
    ################## Check and read new data
    ingested_files = read_ingested_files()
    source_files = discover_source_files()
    new_files = sorted(set(source_files) - set(ingested_files))

    ################## Deciding whether to proceed, part 1
    if not new_files:
        print('No new data found. Ending process.')
        return

    print(f'New data found: {new_files}. Ingesting.')
    subprocess.run([sys.executable, 'ingestion.py'], check=True)

    ################## Checking for model drift
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        deployed_score = float(f.read().strip())

    drift_score = score_deployed_model_on_new_data()
    print(f'Deployed model F1 on record: {deployed_score:.6f} | on new data: {drift_score:.6f}')

    ################## Deciding whether to proceed, part 2
    if drift_score >= deployed_score:
        print('No model drift detected. Ending process.')
        return

    ################## Re-training a candidate model
    print('Model drift detected. Training a candidate model on the new data.')
    subprocess.run([sys.executable, 'training.py'], check=True)
    subprocess.run([sys.executable, 'scoring.py'], check=True)

    with open(os.path.join(output_model_path, 'latestscore.txt'), 'r') as f:
        candidate_score = float(f.read().strip())

    ################## Deciding whether to deploy: only if the candidate is better
    print(f'Candidate model F1: {candidate_score:.6f} | Deployed model F1: {deployed_score:.6f}')
    if candidate_score <= deployed_score:
        print('Candidate model does not outperform the deployed model. '
              'Deployment skipped; the current model remains in production.')
        return

    ################## Re-deployment
    deployment.store_model_into_pickle()
    print('Candidate model outperforms the deployed model. Re-deployed to production.')

    ################## Diagnostics and reporting
    run_reporting_and_api_calls()
    print('Second-run reports generated: confusionmatrix2.png, apireturns2.txt.')


if __name__ == '__main__':
    main()
