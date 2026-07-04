import json
import os

import requests


with open('config.json', 'r') as f:
    config = json.load(f)

# Specify a URL that resolves to your workspace
URL = 'http://127.0.0.1:8000'
output_model_path = os.path.join(config['output_model_path'])


# Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction', json={'filepath': 'testdata/testdata.csv'})
response2 = requests.get(f'{URL}/scoring')
response3 = requests.get(f'{URL}/summarystats')
response4 = requests.get(f'{URL}/diagnostics')

# Combine all API responses
responses = {
    'prediction': response1.json(),
    'scoring': response2.json(),
    'summarystats': response3.json(),
    'diagnostics': response4.json(),
}

# Write the responses to your workspace
os.makedirs(output_model_path, exist_ok=True)
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as f:
    f.write(json.dumps(responses, indent=2))
