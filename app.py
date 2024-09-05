from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application
#route for a homepage
@app.route('/')
def index():
    return render_template('index.html')

"""
@app.route('/predictdata',methods=['Get','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Geography=request.form.get('Geography'),
            CreditScore=request.form.get('CreditScore'),
            Age=request.form.get('Age'),
            Tenure=request.form.get('Tenure'),
            Balance=float(request.form.get('Balance')),
            EstimatedSalary=float(request.form.get('EstimatedSalary'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        

        predict_pipeline=PredictPipeline()
      
        results=predict_pipeline.predict(pred_df)
    
        return render_template('home.html',results=results[0])
    
"""
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        