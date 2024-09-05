from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect data from form input
        data = CustomData(
            Geography=request.form['Geography'],
            Gender=request.form['Gender'],
            CreditScore=float(request.form['CreditScore']),
            Age=int(request.form['Age']),
            Tenure=int(request.form['Tenure']),
            Balance=float(request.form['Balance']),
            NumOfProducts=int(request.form['NumOfProducts']),
            HasCrCard=int(request.form['HasCrCard']),
            IsActiveMember=int(request.form['IsActiveMember']),
            EstimatedSalary=float(request.form['EstimatedSalary'])
        )

        # Convert the input data to a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        # Load prediction pipeline and predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
    
        # Interpret the prediction result
        if results[0] == 1:
            result = "Exited"
        else:
            result = "Not Exited"

        # Return the prediction result to the index page with a message
        return render_template('home.html', prediction_text=f'Prediction: {result}')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
