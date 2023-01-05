import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



# ------------------- TRAIN, VALIDATE, TEST -------------------

def my_train_test_split(telco, churn_Yes):
    
    train, test = train_test_split(telco, test_size=.2, random_state=123, stratify=telco[churn_Yes])
    
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[churn_Yes])
    
    return train, validate, test



# ------------------- TELCO DATA -------------------

def split_telco_data(telco):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(telco, test_size=.2, 
                                        random_state=123, 
                                        stratify=telco.churn_Yes)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn_Yes)
    return train, validate, test

def prep_telco(telco):
    '''
    This function will prepare the telco_churn dataset for exploration
    '''
    # Convert to correct datatype 
    telco['total_charges'] = telco['total_charges'].replace(' ', '0')
    telco['total_charges'] = telco['total_charges'].astype(float)
    
    
    to_dummy = ['churn', 'gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 
                'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
                'paperless_billing', 'contract_type', 'internet_service_type', 
                'payment_type']
    # Create dummies for non-binart categorical variables
    dummies = pd.get_dummies(telco[to_dummy], drop_first=False)
    telco = pd.concat([telco, dummies], axis=1)

    # Drop duplicate columns
    drop = ['multiple_lines_No phone service', 'online_security_No internet service',
        'online_backup_No internet service', 'device_protection_No internet service',
        'tech_support_No internet service', 'streaming_tv_No internet service',
        'streaming_movies_No internet service', 'gender_Female', 'partner_No',
        'dependents_No', 'phone_service_No', 'multiple_lines_No', 'online_security_No', 'online_backup_No',
        'device_protection_No', 'tech_support_No', 'streaming_tv_No',
        'streaming_movies_No', 'paperless_billing_No', 'gender', 'partner', 
        'dependents', 'phone_service', 'multiple_lines', 'online_security', 
        'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
        'paperless_billing', 'payment_type', 'payment_type_id', 'internet_service_type_id', 
        'contract_type_id', 'churn', 'churn_No']
    telco.drop(columns=drop, inplace=True)
                 
    return telco

