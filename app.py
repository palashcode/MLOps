from validate import validate_model
from predict import make_prediction
import requests
import os
import pandas as pd
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return {"app_status":"app live!"}

@app.route('/version')
def get_version():
    with open("models/version", "r") as fp:
        version = fp.read()
    return {"model_version":version}

@app.route('/health')
def get_health():
    if validate_model():
        return "good"
    else:
        return "bad"

@app.route('/predict')
def predict():
    data = requests.get(
        "https://raw.githubusercontent.com/palashcode/MLOps/master/data/validation.csv")
    with open("temp.csv", "w") as fp:
        fp.write(data.text)
    data = pd.read_csv("temp.csv")
    result = make_prediction(data)
    res = {}
    for i in range(len(result)):
        res[str(i)] = "converted" if result[i]==1 else "not converted"
    return res
 
if __name__ == "__main__":
    PORT = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=PORT)