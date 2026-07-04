from flask import Flask, jsonify, request
import json
import os

import pandas as pd

import diagnostics
import scoring


###################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])


####################### Prediction Endpoint
@app.route('/prediction', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    request_data = request.get_json(silent=True) or {}
    file_path = request_data.get('filepath') or request.args.get('filepath')

    if not file_path:
        return jsonify({'error': 'filepath is required'}), 200

    data = pd.read_csv(file_path)
    predictions = diagnostics.model_predictions(data)
    return jsonify({'predictions': predictions}), 200


####################### Scoring Endpoint
@app.route('/scoring', methods=['GET', 'OPTIONS'])
def scoring_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    score = scoring.score_model()
    return jsonify({'f1_score': score}), 200


####################### Summary Statistics Endpoint
@app.route('/summarystats', methods=['GET', 'OPTIONS'])
def summarystats_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    summary_statistics = diagnostics.dataframe_summary()
    return jsonify({'summary_statistics': summary_statistics}), 200


####################### Diagnostics Endpoint
@app.route('/diagnostics', methods=['GET', 'OPTIONS'])
def diagnostics_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    execution_times = diagnostics.execution_time()
    missing_percentages = diagnostics.missing_data()
    dependency_table = diagnostics.outdated_packages_list().to_dict(orient='records')

    return jsonify({
        'execution_time': execution_times,
        'missing_data_percent': missing_percentages,
        'outdated_packages': dependency_table,
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
