First step:

`git clone https://github.com/DanielEgdal/LSDA_A3`

`cd LSDA_A3`
____
To run your own version:

`mlflow run .`

To serve this version:

`mlflow models serve -m best_model -h 0.0.0.0`
____
To simply serve without running: 

`cd serving`

`mlflow models serve -m best_model -h 0.0.0.0`

Query of model: http://40.127.101.229:5000/invocations
At https://orkneycloud.itu.dk/mlflow/ for checking.

Test script: `curl http://40.127.101.229:5000/invocations -H 'Content-Type: application/json' -d '{"columns": ["Speed","Direction"],"data":[[7.15264,"NNW"],[3.12928,"W"],[5.81152,"NNW"],[7.15264,"NNW"]]}'`

____
Notes:

* Conda is not working quite correct. Therefore, the conda.yaml file is being oversaved with one that works. An Sklearn library (`/lib/python3.8/site-packages/mlflow/utils/validation.py`) is uploaded slightly differently to not cause errors.

