import json
import os
import shutil


################## Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
ingested_data_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#################### Function for deployment
def store_model_into_pickle(model=None):
    # Copy the latest pickle file, the latestscore.txt value,
    # and the ingestfiles.txt file into the deployment directory
    os.makedirs(prod_deployment_path, exist_ok=True)

    files_to_copy = [
        (os.path.join(model_path, 'trainedmodel.pkl'), os.path.join(prod_deployment_path, 'trainedmodel.pkl')),
        (os.path.join(model_path, 'latestscore.txt'), os.path.join(prod_deployment_path, 'latestscore.txt')),
        (os.path.join(ingested_data_path, 'ingestedfiles.txt'), os.path.join(prod_deployment_path, 'ingestedfiles.txt')),
    ]

    for source, destination in files_to_copy:
        shutil.copy2(source, destination)


if __name__ == '__main__':
    store_model_into_pickle()
