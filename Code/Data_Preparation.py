import numpy as np
import pandas as pd
import joblib

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder


categorical_column=['ZIP_V2', 'ALUMNI_TYPE', 'UG_SCHOOL', 'GRAD_DEGREE', 'GRAD_CLASS_YEAR', 'GRAD_SCHOOL', 'UG_CLASS_YEAR','STATE','CITY']

def inspect_data(file_path):
    
    df=pd.read_csv(file_path)
    df.info()
    df.describe()
    df.head()
    df.dtypes
    df.isnull().sum()



def year_mapping(year):
        if np.isnan(year):
            return 'NA'
        else:
            return str(int(year//10 * 10)) + 's'



def encode_columns(encoder,X):

    encoded_columns = encoder.transform(X[categorical_column])
    encoded_columns = encoded_columns.rename(columns = {'ZIP_V2': 'ZIP_ENCODED', 'ALUMNI_TYPE': 'ALUMNI_TYPE_ENCODED', 'UG_SCHOOL': 'UG_SCHOOL_ENCODED', 'GRAD_DEGREE': 'GRAD_DEGREE_ENCODED', 'GRAD_CLASS_YEAR': 'GRAD_CLASS_YEAR_ENCODED', 'GRAD_SCHOOL': 'GRAD_SCHOOL_ENCODED', 'UG_CLASS_YEAR': 'UG_CLASS_YEAR_ENCODED','STATE': 'STATE_ENCODED', 'CITY': 'CITY_ENCODED'})

    X_encoded = pd.concat([X, pd.DataFrame(encoded_columns)], axis = 1)
    X_encoded = X_encoded.drop(['ZIP_V2', 'ALUMNI_TYPE', 'UG_SCHOOL', 'GRAD_DEGREE', 'GRAD_CLASS_YEAR', 'GRAD_SCHOOL', 'UG_CLASS_YEAR','STATE','CITY'], axis = 1)
    
    return X_encoded



def load_encoder():
    model_name='DonationMaximization_MTL_Model'
    artifact_path='encoder.pkl'
    client= MlflowClient()
    production_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    run_id = production_version.run_id
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    encoder = joblib.load(path)
    return encoder



def prepare_data(file_path,retrain):
    df=pd.read_csv(file_path)
    
    df=df.drop(['MI'], axis = 1)
    
    df['ZIP_V2'] = df['ZIP_V2'].fillna(value = '000000')
    df['ZIP_V2'] = df['ZIP_V2'].apply(lambda x: '000000' if len(str(x)) < 6 else x)
    df['ZIP_V2'] = df['ZIP_V2'].fillna(value = '000000')
    
    GRAD_SCHOOL_mapping_dict = {
    'College of Arts and Sci (Sci)': 'Arts and Sciences',
    'College of Arts and Sci (Arts)': 'Arts and Sciences',
    'Col of Arts/Sci and Sch Mgt': 'Arts and Sciences',
    'School of Law': 'Law',
    'School of Education': 'Education',
    'School of Nursing': 'Nursing',
    'School of Business': 'Business and Professional Studies',
    'Sch of Bus and Prof Studies': 'Business and Professional Studies',
    'College of Prof. Studies': 'Business and Professional Studies',
    np.nan: 'GRAD_SCHOOL_Unknown'
}
    df['GRAD_SCHOOL'] = df['GRAD_SCHOOL'].replace(GRAD_SCHOOL_mapping_dict)
    
    UG_SCHOOL_mapping_dict = {
    'School of Nursing': 'School of Nursing',
    'School of Business': 'Business and Professional Studies',
    'Sch of Bus and Prof Studies': 'Business and Professional Studies',
    'College of Arts and Sci (Arts)': 'Arts and Sciences',
    'College of Arts and Sci (Sci)': 'Arts and Sciences',
    'School of Education': 'School of Education',
    'College of Prof. Studies': 'Business and Professional Studies',
    'School of Law': 'School of Law',
    'Continuing Education': 'Continuing Education',
    'Not used in standing': 'Not Categorized',
    np.nan: 'UG_SCHOOL_Unknown'
    }
    df['UG_SCHOOL'] = df['UG_SCHOOL'].replace(UG_SCHOOL_mapping_dict)
    
    GENDER_mapping_dict = {
    'M': 1,
    'F': 0,
    np.nan: 'GENDER_Unknown'
    }
    MARRIED_TO_ALUM_mapping_dict = {
    'N': 0,
    'Y': 1,
    np.nan: 'MARRIED_TO_ALUM_Unknown'
    }
    df['GENDER'] = df['GENDER'].replace(GENDER_mapping_dict)
    df['MARRIED_TO_ALUM'] = df['MARRIED_TO_ALUM'].replace(MARRIED_TO_ALUM_mapping_dict)
    
    df['UG_CLASS_YEAR'] = df['UG_CLASS_YEAR'].apply(year_mapping)
    df['GRAD_CLASS_YEAR'] = df['GRAD_CLASS_YEAR'].apply(year_mapping)
    
    df['GRAD_DEGREE'] = df['GRAD_DEGREE'].fillna(value = 'NA')

    df['STATE'].fillna('NA', inplace=True)
    df['CITY'].fillna('NA', inplace=True)
    df['AVERAGE_DONATIONS'].fillna(0, inplace=True)
    
    if retrain==True:
        df['DONATED'] = df['NUMBER_OF_DONATIONS'].apply(lambda x: 1 if x > 0 else 0)
        df['AVERAGE_DONATIONS'] = df['VALUE_OF_DONATIONS'] / df['NUMBER_OF_DONATIONS']
        
        X = df.drop(['ZIP','DONATED', 'NUMBER_OF_DONATIONS', 'VALUE_OF_DONATIONS', 'ACCOUNT_ID', 'AVERAGE_DONATIONS'], axis = 1)
        Y_regression = df['VALUE_OF_DONATIONS']
        Y_classification= df['DONATED']
        
        X_train_val, X_test, Y_regression_train_val, Y_regression_test, Y_classification_train_val, Y_classification_test = train_test_split(X, Y_regression, Y_classification, test_size=0.3, random_state=30)
        X_train, X_val, Y_regression_train, Y_regression_val, Y_classification_train, Y_classification_val = train_test_split(X_train_val, Y_regression_train_val, Y_classification_train_val, test_size=0.25, random_state=30)
        
        target_encoder = TargetEncoder()
        encoder=target_encoder.fit(X_train[categorical_column], Y_classification_train)
        
        joblib.dump(encoder, "encoder.pkl")
        mlflow.log_artifact("encoder.pkl")
        
        
        
        X_train_encoded=encode_columns(encoder,X_train,Y_classification_train)
        X_val_encoded=encode_columns(encoder,X_val,Y_classification_val)
        X_test_encoded=encode_columns(encoder,X_test,Y_classification_test)
        
        data_dict = {
            'X_train': X_train_encoded,
            'Y_regression_train': Y_regression_train,
            'Y_classification_train': Y_classification_train,
            'X_val': X_val_encoded,
            'Y_regression_val': Y_regression_val,
            'Y_classification_val': Y_classification_val,
            'X_test': X_test_encoded,
            'Y_regression_test': Y_regression_test,
            'Y_classification_test': Y_classification_test
            }
        
        return data_dict
    
        
    else:
        X = df.drop(['ZIP','DONATED', 'NUMBER_OF_DONATIONS', 'VALUE_OF_DONATIONS', 'ACCOUNT_ID', 'AVERAGE_DONATIONS'], axis = 1)
        
        encoder=load_encoder()
        X_encoded=encode_columns(encoder,X)
    
        data_dict = {
            'X_test':X_encoded
            }
        
        return data_dict