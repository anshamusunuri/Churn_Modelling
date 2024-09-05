import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            #model_path="artifacts/model.pkl"
            #preprocessor_path="artifacts/preprocessor.pkl"
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            
            
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
          
            
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        Gender: str,
        Geography: str,
        CreditScore:int,
        Age: int,
        Tenure: int,
        Balance: int,
        EstimatedSalary: int):

        self.Gender = Gender
        self.Geography = Geography
        self.CreditScore = CreditScore
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.EstimatedSalary = EstimatedSalary

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Geography": [self.Geography],
                "CreditScore": [self.CreditScore],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
                "EstimatedSalary": [self.EstimatedSalary],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)