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

* The file with_logging_basic.py is not in use because of issues in the an mlflow file. If you have the file on your local machine, go to `[...]lib/python3.8/site-packages/mlflow/utils/validation.py` and change all instances of `250` to `500` 

