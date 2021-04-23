First step:

`git clone https://github.com/DanielEgdal/LSDA_A3`

`cd LSDA_A3`
____
## Run your own version:

`mlflow run .`

To **serve** this version:

`mlflow models serve -m best_model -h 0.0.0.0`
____
## Serve without running: 

`cd serving`

`mlflow models serve -m best_model -h 0.0.0.0`

Query of model: http://40.127.101.229:5000/invocations
At https://orkneycloud.itu.dk/mlflow/ for checking.

Test script: `curl http://40.127.101.229:5000/invocations -H 'Content-Type: application/json' -d '{"columns": ["Speed","Direction"],"data":[[7.15264,"NNW"],[3.12928,"W"],[5.81152,"NNW"],[7.15264,"NNW"]]}'`

____
## Automatic retraining:
Create a Bash file, example:

`#! /bin/bash`

`export MLFLOW_CONDA_HOME=/home/daeg/anaconda3`

`killall -9 gunicorn`
`killall -9 mlflow`

`rm -rf /home/daeg/a3/git3/LSDA_A3/best_model`

`/home/daeg/anaconda3/bin/mlflow run /home/daeg/a3/git3/LSDA_A3 >> /home/daeg/cron.log 2>> /home/daeg/cron.error`

`/home/daeg/anaconda3/bin/mlflow models serve -m /home/daeg/a3/git3/LSDA_A3/best_model -h 0.0.0.0 >> /home/daeg/cron.log 2>> /home/daeg/cron.error`

Create cronjob: 

`crontab -e`

`0 */12 * * * /home/daeg/a3/git3/retrain.sh`

Notes:

* Conda is not working quite correct. Therefore, the conda.yaml file is being oversaved with one that works. An Sklearn library (`/lib/python3.8/site-packages/mlflow/utils/validation.py`) is uploaded slightly differently to not cause errors.

