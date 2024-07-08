import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pickle

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)

def eval_models(X_train,y_train,X_test,y_test,models):
    try:
        report ={}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            print(f'{list(models.keys())[i]} training compeleted')
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test,y_pred)
            report[list(models.keys())[i]] = r2
        return report
    except Exception as e:
        logging.info('Exception occured during eval_models')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)