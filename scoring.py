import json
import os
import pickle

import pandas as pd
from sklearn.metrics import f1_score


################# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


################# Function for model scoring
def score_model():
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_data['exited']

    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)

    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1))

    return f1


if __name__ == '__main__':
    score_model()
