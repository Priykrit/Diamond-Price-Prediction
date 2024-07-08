import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    data_ingestion_obj=DataIngestion()
    train_data_path,test_data_path = data_ingestion_obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    data_transformation_obj = DataTransformation()
    (train_arr,test_arr,preporcessor_obj_file_path)=data_transformation_obj.initiate_data_transformation(train_data_path,test_data_path)
    print(preporcessor_obj_file_path)
    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_training(train_arr,test_arr)