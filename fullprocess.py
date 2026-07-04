import ast
import json
import os
import shutil
import subprocess
import sys
import time

import deployment
import reporting
import scoring


with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']


def read_ingested_files():
    ingested_file_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    if not os.path.exists(ingested_file_path):
        return []

    with open(ingested_file_path, 'r') as f:
        content = f.read().strip()

    if not content:
        return []

    return ast.literal_eval(content)


def discover_source_files():
    return sorted(
        file_name
        for file_name in os.listdir(input_folder_path)
        if file_name.endswith('.csv')
    )


def start_api_server():
    server = subprocess.Popen(
        [sys.executable, 'app.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return server


def run_reporting_and_api_calls():
    server = start_api_server()
    try:
        subprocess.run([sys.executable, 'apicalls.py'], check=True)
    finally:
        server.terminate()
        server.wait(timeout=10)

    reporting.score_model()


def archive_submission_outputs():
    confusion_matrix_path = os.path.join(output_model_path, 'confusionmatrix.png')
    api_returns_path = os.path.join(output_model_path, 'apireturns.txt')

    if os.path.exists(confusion_matrix_path):
        shutil.copy2(
            confusion_matrix_path,
            os.path.join(output_model_path, 'confusionmatrix2.png'),
        )

    if os.path.exists(api_returns_path):
        shutil.copy2(
            api_returns_path,
            os.path.join(output_model_path, 'apireturns2.txt'),
        )


def main():
    ################## Check and read new data
    ingested_files = read_ingested_files()
    source_files = discover_source_files()
    new_files = sorted(set(source_files) - set(ingested_files))

    ################## Deciding whether to proceed, part 1
    if not new_files:
        print('No new data found. Ending process.')
        return

    subprocess.run([sys.executable, 'ingestion.py'], check=True)

    ################## Checking for model drift
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        deployed_score = float(f.read().strip())

    subprocess.run([sys.executable, 'training.py'], check=True)
    new_score = scoring.score_model()

    ################## Deciding whether to proceed, part 2
    if new_score >= deployed_score:
        print('No model drift detected. Ending process.')
        return

    ################## Re-deployment
    deployment.store_model_into_pickle()

    ################## Diagnostics and reporting
    run_reporting_and_api_calls()
    archive_submission_outputs()
    print('Model drift detected. Pipeline completed.')


if __name__ == '__main__':
    main()
