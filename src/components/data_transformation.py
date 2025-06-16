import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            logging.info(f'Numerical features: {numerical_features}')
            logging.info(f'Categorical features: {categorical_features}')
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numerical_pipeline, numerical_features),
                    ('cat_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            logging.info("Data transformation object created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error occurred while creating data transformation object")
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Loaded train and test data successfully')
            
            logging.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_data_transformer_object()
            target_column = 'math score'
            numerical_features = ['reading score', 'writing score']
            
            target_feature_train = train_df[target_column]
            input_features_train = train_df.drop(columns=[target_column], axis=1)
            
            target_feature_test = test_df[target_column]
            input_features_test = test_df.drop(columns=[target_column], axis=1)
            
            logging.info('Applying preprocessing object on training and testing data')
            
            input_features_train_transformed = preprocessor_obj.fit_transform(input_features_train)
            input_features_test_transformed = preprocessor_obj.transform(input_features_test)
            
            train_arr = np.c_[input_features_train_transformed, np.array(target_feature_train)]
            test_arr = np.c_[input_features_test_transformed, np.array(target_feature_test)]
            logging.info(f'Saved preprocessor object to {self.data_transformation_config.preprocessor_obj_file_path}')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
                       
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys) from e
