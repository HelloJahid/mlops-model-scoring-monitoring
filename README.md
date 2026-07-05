# MLOps Model Scoring and Monitoring

A dynamic risk assessment system that predicts corporate client attrition and keeps itself
accurate through automated re-training, re-deployment, monitoring, and reporting.

Udacity Machine Learning DevOps Engineer — Project: ML Pipeline Workflow including
Monitoring and Scoring.

## Pipeline Overview

| Step | Script | Output |
|------|--------|--------|
| 1. Data ingestion | `ingestion.py` | `ingesteddata/finaldata.csv`, `ingesteddata/ingestedfiles.txt` |
| 2. Training | `training.py` | `models/trainedmodel.pkl` |
| 2. Scoring | `scoring.py` | `models/latestscore.txt` (F1 on test data) |
| 2. Deployment | `deployment.py` | copies model, score, ingestion record to `production_deployment/` |
| 3. Diagnostics | `diagnostics.py` | predictions, summary stats, NA %, timings, dependency check |
| 4. Reporting | `reporting.py` | `models/confusionmatrix.png` |
| 4. API | `app.py` (+ `wsgi.py`) | `/prediction`, `/scoring`, `/summarystats`, `/diagnostics` on port 8000 |
| 4. API client | `apicalls.py` | `models/apireturns.txt` |
| 5. Automation | `fullprocess.py` + `cronjob.txt` | re-deployment decision process every 10 minutes |

## Configuration

All paths are read from `config.json`. Development uses `practicedata` / `practicemodels`;
production uses `sourcedata` / `models` (current setting).

## How to Run

```bash
pip install -r requirements.txt

# Run individual steps
python ingestion.py
python training.py
python scoring.py
python deployment.py
python diagnostics.py
python reporting.py

# API: start the server, then call the endpoints
python app.py            # terminal 1
python apicalls.py       # terminal 2

# Full automated process
python fullprocess.py
```

## Automation Logic (`fullprocess.py`)

1. Compare files in `input_folder_path` against `production_deployment/ingestedfiles.txt`.
   If no new files exist, stop.
2. Ingest the new data, then score the currently deployed model on it. If the F1 on the new
   data is greater than or equal to the previously recorded score, no model drift has
   occurred, so stop.
3. Otherwise train a candidate model on the new data and score it on the test set.
4. Deploy the candidate only if it outperforms the currently deployed model. A candidate
   that scores worse is never deployed, so production performance can only improve.
5. After a successful re-deployment, regenerate the confusion matrix and API reports for
   the new model as `confusionmatrix2.png` and `apireturns2.txt`.

The initial production deployment is bootstrapped by running the full pipeline manually
after switching `config.json` to `sourcedata` and `models`, following the project
instructions. All subsequent automated re-deployments are gated by the score comparison
above. The first-run reports (`confusionmatrix.png`, `apireturns.txt`) are produced by the
practice configuration run and the second-run reports (`confusionmatrix2.png`,
`apireturns2.txt`) are produced directly by the production configuration run, so the two
pairs are distinct artefacts rather than copies.

## Cron Job

```bash
service cron start
crontab -e   # add the line from cronjob.txt, adjusting the repository path
crontab -l   # verify
```

`cronjob.txt` runs `fullprocess.py` every 10 minutes.

## Submission Artefacts

Practice phase (`practicemodels/`): `trainedmodel.pkl`, `latestscore.txt`,
`confusionmatrix.png`, `apireturns.txt`.
Production phase (`models/`): the same set plus `confusionmatrix2.png` and `apireturns2.txt`
generated after switching `config.json` to `sourcedata` / `models`.
