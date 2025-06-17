import sys
import os
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
    
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    
    model_report = {}
    
    for model_name, model in models.items():
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)
        
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
            logging.info(f"Object loaded successfully from {file_path}")
            return obj
            
    except Exception as e:
        logging.error(f"Error occurred while loading object: {e}")
        raise CustomException(e, sys) from e