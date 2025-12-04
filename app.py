from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

# route to display the home page
@app.route('/', methods=['GET'])  
def homePage():
    return render_template("index.html")

# route to train the pipeline
@app.route('/train', methods=['GET'])  
def training():
    os.system("python main.py")
    return "Training Successful!" 

# dictionary of valid input ranges for each feature
# Values obtained from experimentation in research/input_validation.ipynb using IQR method
valid_input_ranges = {
    'fixed_acidity': [3.95, 12.35],
    'volatile_acidity': [0.02, 1.02],
    'citric_acid': [0.0, 0.91],
    'residual_sugar': [0.85, 3.65],
    'chlorides': [0.04, 0.12],
    'free_sulfur_dioxide': [1.0, 42.0],
    'total_sulfur_dioxide': [6.0, 122.0],
    'density': [0.99, 1.0],
    'pH': [2.92, 3.68],
    'sulphates': [0.28, 1.0],
    'alcohol': [7.1, 13.5]
}

# route to show the predictions in a web UI
@app.route('/predict', methods=['POST','GET']) 
def index():
    if request.method == 'POST':
        try:
            # read inputs from the form
            inputs = {
                'fixed_acidity': float(request.form['fixed_acidity']),
                'volatile_acidity': float(request.form['volatile_acidity']),
                'citric_acid': float(request.form['citric_acid']),
                'residual_sugar': float(request.form['residual_sugar']),
                'chlorides': float(request.form['chlorides']),
                'free_sulfur_dioxide': float(request.form['free_sulfur_dioxide']),
                'total_sulfur_dioxide': float(request.form['total_sulfur_dioxide']),
                'density': float(request.form['density']),
                'pH': float(request.form['pH']),
                'sulphates': float(request.form['sulphates']),
                'alcohol': float(request.form['alcohol'])
            }

            # validate and clip inputs to acceptable range
            for feature, value in inputs.items():
                min_val, max_val = valid_input_ranges[feature]
                if value < min_val:
                    inputs[feature] = min_val
                elif value > max_val:
                    inputs[feature] = max_val

            # convert to array for prediction
            data = np.array(list(inputs.values())).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong. Please check your inputs.'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)