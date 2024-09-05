#model_trainer.py - author_Aneesha
#model_trainer.py - code does the model training (always suggested to try all techniques)

import os
import sys
from dataclasses import dataclass


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,f1_score

from sklearn.utils.class_weight import compute_class_weight


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'SVC': SVC(random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42)

                
            }
            params={
                    'Random Forest': {
                        'n_estimators': [100, 200, 500],
                        'max_depth': [10, 20, 30, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2']
                    },
                    'SVC': {
                        'C': [0.1, 1, 10, 100],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale']
                     },
                    'Logistic Regression': {
                         'C': [0.01, 0.1, 1, 10, 100],
                         'solver': ['liblinear', 'lbfgs']
    
            }
                
            }
            logging.info("model is tuning")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"Best Model: {best_model}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted, average='weighted')
            report = classification_report(y_test, predicted)
           
            logging.info(f"Test Accuracy: {accuracy}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"Classification Report:\n{report}")

            return accuracy, f1, report
            
        except Exception as e:
            raise CustomException(e,sys)