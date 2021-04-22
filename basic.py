import mlflow
import mlflow.sklearn
# mlflow.autolog() 
# import logging
# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)
# mlflow.autolog()

from influxdb import InfluxDBClient
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import make_column_transformer
import sklearn.metrics
import joblib
import os
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index
    return df

def interpolate(data):
    cut = 0
    for i,val in enumerate(data.values):
        if type(val[0]) == str and type(val[1]) == float and type(val[2]) == float:
            cut = i
            break
    data_cut = data[cut:] # Remove first missing values, so interpolation is possible
    data_int = data_cut.interpolate(method="polynomial", order=2)
    data_cut2 = data_int.dropna(thresh=2) # Remove columns with no data from forecasts after interpolation
    data_ffill = data_cut2.fillna(method='ffill') # Fill the directions which are missing
    return data_ffill

def load():
    client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
    client.switch_database('orkney')
    # Get the last 90 days of power generation data
    generation = client.query(
    "SELECT time,Total FROM Generation where time > now()-90d")
    
    # Get the last 90 days of weather forecasts
    wind  = client.query(
    "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'")
    
    # Get forcasts
    forecasts  = client.query(
    "SELECT * FROM MetForecasts where time > now()")
    # Send data to DFs
    gen_df = get_df(generation)
    wind_df = get_df(wind)
    for_df = get_df(forecasts)
    
    # Join tables
    train_df = gen_df.join(wind_df)[['Direction','Speed','Total']]
    
    train_df=train_df.drop_duplicates()
    train_df_int = interpolate(train_df) # Fixing NaNs
    
    newest_source_time = for_df["Source_time"].max()
    newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time][['Direction','Speed']].copy()
    
    # splitting to x and y
    train_x = train_df_int[['Direction','Speed']]
    train_Y = train_df_int[['Total']]
    return train_x,train_Y,newest_forecasts

def train_dev_splitc(x,y): 
    # Potentially coul have added a seperate test, but as the data is 
    # constantly changing, and the last values matters most, 
    # the dev set is used for the final evaluation
    split_cons = len(x) - (len(x)//8)
    x,xdev,y,ydev = x[:split_cons],x[split_cons:],y[:split_cons],y[split_cons:]
    return x,xdev,y,ydev

def extended_pipe_grad(): # Gradient booster pipeline
    short_array = [np.array(['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
    'SSW', 'SW', 'W', 'WNW', 'WSW'], dtype=object)]
    ct = make_column_transformer(
        (OneHotEncoder(categories = short_array),[0]),
        (StandardScaler(), [1])
        )
    pipe2 = Pipeline(
    [('encode',ct),
    ('grad_reg', GradientBoostingRegressor())]
    )
    return pipe2

def extended_pipe_dtree(): # Decision tree pipeline
    short_array = [np.array(['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
    'SSW', 'SW', 'W', 'WNW', 'WSW'], dtype=object)]
    ct = make_column_transformer(
        (OneHotEncoder(categories = short_array),[0]),
        (StandardScaler(), [1])
        )
    pipe2 = Pipeline(
    [('encode',ct),
    ('dtree_reg', DecisionTreeRegressor())]
    )
    return pipe2

def run_pipe(x,y,pipeli,which1): # fitting
    pipe = pipeli
    tscv = TimeSeriesSplit()
    if which1 == 1:
        gparams = {'grad_reg': [GradientBoostingRegressor()], 
        'grad_reg__n_estimators':[75,100,125],
        'grad_reg__max_depth':[2,3,4],
        'grad_reg__min_samples_leaf':[1,2,3]}
        grid = GridSearchCV(pipeli,gparams,cv = tscv, verbose = 10)
        grid.fit(x,y)
        # pipe.fit(x,y)
        # print(pipe[0].transformers_[0][1].categories_)
        print("grad boost grid")
        return grid.best_estimator_
    if which1 == 2:
        dparams = {'dtree_reg': [DecisionTreeRegressor()], 
        'dtree_reg__max_depth':[4,5,6,7,8],
        'dtree_reg__min_samples_split':[1,2,4,6],
        'dtree_reg__min_samples_leaf':[1,2,3]}
        grid = GridSearchCV(pipeli,dparams,cv = tscv)
        grid.fit(x,y)
        print("dtree grid")
        # print(dir(grid))
        return grid.best_estimator_

def get_dev_scores(xdev,ydev,pipe): # MSE score
    mse_score = sklearn.metrics.mean_squared_error(ydev,pipe.predict(xdev))
    return mse_score

def dtree_or_grad_boost(xdev,ydev,dtree,grad_boost): # Choose either decision tree or gradient booster
    with mlflow.start_run(nested= True):
        dscore = get_dev_scores(xdev,ydev,dtree)
        mlflow.log_metric("MSE",dscore)
        mlflow.log_params(dtree.get_params())
        mlflow.sklearn.log_model(sk_model=dtree,artifact_path="Decision tree")
        # mlflow.sklearn.save_model(sk_model=dtree,path="Decision tree")
    with mlflow.start_run(nested= True):
        gbscore = get_dev_scores(xdev,ydev,grad_boost)
        mlflow.log_metric("MSE",gbscore)
        mlflow.log_params(grad_boost.get_params())
        mlflow.sklearn.log_model(sk_model=grad_boost,artifact_path="Gradient Booster")
        # mlflow.sklearn.save_model(sk_model=grad_boost,path="Gradient Booster")
    print("Decision tree score:",dscore,"Gradient Booster score:",gbscore)
    if dscore < gbscore:
        print("Decision tree trained, checking if better than old model")
        return dtree
    else:
        print("Gradient booster trained, checking if better than old model")
        return grad_boost

def old_best(): # Load best model from earlies
    best = joblib.load("best.pkl")
    return best

def save_best(old,new,xdev,ydev): # If the new model is better, oversave the old
    new_score = get_dev_scores(xdev,ydev,new)
    old_score = get_dev_scores(xdev,ydev,old)
    mlflow.log_metric("MSE",new_score)
    if new_score < old_score:
        joblib.dump(new,"best.pkl")
        print('Model oversaved')
    else:
        print('Old model used')

def run_all(Old_mod_pres=True): # Run all together, with the option to not use a previous model
    all_x,all_y,newest_forecasts = load()
    x,xdev,y,ydev=train_dev_splitc(all_x,all_y)
    #with mlflow.start_run():
    grad_pipe = run_pipe(x,y,extended_pipe_grad(),1)
    #grad_pipe = run_pipe(x,y,extended_pipe_dtree(),2)
    dtree_pipe = run_pipe(x,y,extended_pipe_dtree(),2)
    
    new_pipe = dtree_or_grad_boost(xdev,ydev,dtree_pipe,grad_pipe)

    if Old_mod_pres:
        best = old_best()
    
        save_best(best,new_pipe,xdev,ydev)
    else:
        joblib.dump(new_pipe,"best.pkl")
    
    best = old_best() # If the new model was better, load it as the past file was oversaved.
    
    print("Development MSE:",get_dev_scores(xdev,ydev,best))
    mlflow.log_params(new_pipe.get_params())
    mlflow.sklearn.log_model(sk_model=new_pipe,artifact_path="best_model")
    mlflow.sklearn.save_model(sk_model=new_pipe,path="best_model")
    
    return best.predict(newest_forecasts)

# with mlflow.start_run():
#     print(run_all(False))

with mlflow.start_run():
    print(run_all())