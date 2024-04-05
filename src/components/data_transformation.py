import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 
import os 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns_indices = [0, 3, 8, 9, 10]  # Assuming these are the indices of numerical columns
            categorical_columns_indices = [1, 2, 4, 5, 6, 7, 11]  # Assuming these are the indices of categorical columns

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot_encoding', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns_indices),
                ('cat_pipeline', cat_pipeline, categorical_columns_indices)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def handle_missing_values(self, df):
        # Handle '?' values in numerical columns by replacing them with NaN and then filling with median
        numerical_columns_indices = [0, 3, 8, 9, 10]  # Assuming these are the indices of numerical columns
        for col_idx in numerical_columns_indices:
            df.iloc[:, col_idx].replace('?', np.nan, inplace=True)  # Replace '?' with NaN
            df.iloc[:, col_idx] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')  # Convert to numeric type
            median_value = df.iloc[:, col_idx].median()
            df.iloc[:, col_idx].fillna(median_value, inplace=True)  # Fill NaN with median

        # Handle '?' values in categorical columns by replacing them with most frequent value
        categorical_columns_indices = [1, 2, 4, 5, 6, 7, 11]  # Assuming these are the indices of categorical columns
        for col_idx in categorical_columns_indices:
            df.iloc[:, col_idx].replace('?', df.iloc[:, col_idx].mode()[0], inplace=True)  # Replace '?' with most frequent value

        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Handle missing values in both training and testing dataframes
            train_df = self.handle_missing_values(train_df)
            test_df = self.handle_missing_values(test_df)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(train_df)
            input_feature_test_arr = preprocessing_obj.transform(test_df)

            logging.info("Saved preprocessing object")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            return input_feature_train_arr, test_df  # Return the transformed training data and the original test dataframe
        except Exception as e:
            raise CustomException(e, sys)
