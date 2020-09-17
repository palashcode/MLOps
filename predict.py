import pandas as pd
import numpy as np
import joblib
import config
import os

def make_prediction(input_data):
    #load latest model
    model_list = os.listdir("models")
    model_list = sorted(model_list, key = timestamp)
    model_name = "models/"+model_list[-1]
    model = joblib.load(filename=model_name)
    
    results = model.predict(input_data)
    return results

def timestamp(filename):
    return filename.split(".")[0]
   
if __name__ == '__main__':
    
    # test pipeline
    data = pd.read_csv(config.VALIDATION_DATA_FILE)
    data = data.drop(['index', 'closed'], axis=1)
    pred = make_prediction(data)

    print(pred)
