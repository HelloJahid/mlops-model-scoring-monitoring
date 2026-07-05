#!/bin/bash
# Regenerates every submission artefact from scratch in two config phases.
# Run from the repository root:  bash regenerate.sh
set -e

echo "==== Cleaning previous outputs ===="
rm -f ingesteddata/* practicemodels/* models/* production_deployment/*

echo "==== PHASE 1: practice configuration ===="
python - <<'PYEOF'
import json
c = json.load(open('config.json'))
c['input_folder_path'] = 'practicedata'
c['output_model_path'] = 'practicemodels'
json.dump(c, open('config.json','w'), indent=4)
PYEOF
python ingestion.py
python training.py
python scoring.py
python deployment.py
python reporting.py
python app.py > /tmp/flask.log 2>&1 &
APP_PID=$!
sleep 4
python apicalls.py
kill $APP_PID
sleep 1
echo "Practice F1: $(cat practicemodels/latestscore.txt)"

echo "==== PHASE 2: production configuration ===="
python - <<'PYEOF'
import json
c = json.load(open('config.json'))
c['input_folder_path'] = 'sourcedata'
c['output_model_path'] = 'models'
json.dump(c, open('config.json','w'), indent=4)
PYEOF
python ingestion.py
python training.py
python scoring.py
python deployment.py
python reporting.py confusionmatrix2.png
python app.py > /tmp/flask.log 2>&1 &
APP_PID=$!
sleep 4
python apicalls.py apireturns2.txt
kill $APP_PID
sleep 1
echo "Production F1: $(cat models/latestscore.txt)"

echo "==== Sanity: fullprocess should find no new data ===="
python fullprocess.py

echo "==== Done. Artefacts ===="
ls practicemodels/ models/ production_deployment/
