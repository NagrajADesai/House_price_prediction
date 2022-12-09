import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

## starting point
app = Flask(__name__)
## open pickle file in read bite mode
regmodel = pickle.load(open('regmodel.pkl','rb'))
## load scaling file
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
#creat home page
def home():
    return render_template('home.html')

## create predict api
@app.route('/predict_api',methods=['POST'])

def predict_api():
    # take data input in format of json
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    # convert the values as per the scaling
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_dat)
    print(output[0])
    # to convert from 2d array
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)