from env import get_db_url
import numpy as np
import pandas as pd
import os
import acquire
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

def split_scaled_and_unscaled(df, test_size=.2, validate_size=.2, target=None, stratify_col=None):
    if stratify_col==None:
        train_validate, test = train_test_split(df, test_size = .2, random_state=823)
        train, validate = train_test_split(train_validate, test_size=validate_size / (1 - test_size), random_state=823)

        X_train = train.drop(columns=[target])
        y_train = train[target]

        X_validate = validate.drop(columns=[target])
        y_validate = validate[target]

        X_test = test.drop(columns=[target])
        y_test = test[target]

        #Set the Scaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        # Note that we only call .fit with the training data,
        # but we use .transform to apply the scaling to all the data splits.
        scaler.fit(X_train)

        # Turn all the sets inot the scaled np data array
        X_train_scaled = scaler.transform(X_train)
        X_validate_scaled = scaler.transform(X_validate)
        X_test_scaled = scaler.transform(X_test)
        
        # Create a dictionary so that I can take the np arrays back to a labelled pd DataFrame
        columns = train.columns #List of Columns
        numbers = [0,1,2,3,4,5,6] #List of numbers for the scaled np array I'm converting into a dataframe
        zipped= dict(zip(numbers, columns))

        
        #turn the Train Validate and Test arrays back into labelled DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled).rename(columns=zipped)
        X_validate_scaled = pd.DataFrame(X_validate_scaled).rename(columns=zipped)
        X_test_scaled = pd.DataFrame(X_test_scaled).rename(columns=zipped)
        
        #Return the three scaled DataFrames
        return X_train_scaled, X_validate_scaled, X_test_scaled, X_train, y_train, X_validate, y_validate, X_test, y_test
    
    else:
        train_validate, test = train_test_split(df, test_size = .2, random_state=823, stratify=df[stratify_col])
        train, validate = train_test_split(train_validate, test_size=validate_size / (1 - test_size), random_state=823, stratify=train_validate[stratify_col])
    
        X_train = train.drop(columns=[stratify_col])
        y_train = train[stratify_col]

        X_validate = validate.drop(columns=[stratify_col])
        y_validate = validate[stratify_col]

        X_test = test.drop(columns=[stratify_col])
        y_test = test[stratify_col]

        #Set the Scaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        # Note that we only call .fit with the training data,
        # but we use .transform to apply the scaling to all the data splits.
        scaler.fit(X_train)

        # Turn all the sets inot the scaled np data array
        X_train_scaled = scaler.transform(X_train)
        X_validate_scaled = scaler.transform(X_validate)
        X_test_scaled = scaler.transform(X_test)
        
        # Create a dictionary so that I can take the np arrays back to a labelled pd DataFrame
        columns = train.columns #List of Columns
        numbers = [0,1,2,3,4,5,6] #List of numbers for the scaled np array I'm converting into a dataframe
        zipped= dict(zip(numbers, columns))

        
        #turn the Train Validate and Test arrays back into labelled DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled).rename(columns=zipped)
        X_validate_scaled = pd.DataFrame(X_validate_scaled).rename(columns=zipped)
        X_test_scaled = pd.DataFrame(X_test_scaled).rename(columns=zipped)
        
        #Return the three scaled DataFrames
        return X_train_scaled, X_validate_scaled, X_test_scaled, X_train, y_train, X_validate, y_validate, X_test, y_test
    
    

def split_data(df, stratify_col=None):
    if stratify_col == None:
        train_validate, test = train_test_split(df, test_size = .2, random_state=823)
        train, validate = train_test_split(train_validate, test_size=.25, random_state=823)
        return train, validate, test

    else:
        train_validate, test = train_test_split(df, test_size = .2, random_state=823, stratify=df[stratify_col])
        train, validate = train_test_split(train_validate, test_size=.25, random_state=823, stratify=train_validate[stratify_col])
        return train, validate, test

def split_data_label(df, stratify_col, test_size=.2, validate_size=.2):
    train_validate, test = train_test_split(df, test_size = .2, random_state=823, stratify=df[stratify_col])
    train, validate = train_test_split(train_validate, test_size= validate_size / (1 - test_size), random_state=823, stratify=train_validate[stratify_col])
    
    X_train = train.drop(columns=[stratify_col])
    y_train = train[stratify_col]

    X_validate = validate.drop(columns=[stratify_col])
    y_validate = validate[stratify_col]

    X_test = test.drop(columns=[stratify_col])
    y_test = test[stratify_col]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def scale_zillow(zillow):  # doesn't remove the target
    # Split the Data
    train_validate, test = train_test_split(zillow, test_size = .2, random_state=823)
    train, validate = train_test_split(train_validate, test_size= .25, random_state=823)
    
    #Set the Scaler
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(train)

    # Turn all the sets inot the scaled np data array
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # Create a dictionary so that I can take the np arrays back to a labelled pd DataFrame
    columns = train.columns #List of Columns
    numbers = [0,1,2,3,4,5,6] #List of numbers for the scaled np array I'm converting into a dataframe
    zipped= dict(zip(numbers, columns))

    
    #turn the Train Validate and Test arrays back into labelled DataFrames
    train_scaled = pd.DataFrame(train_scaled).rename(columns=zipped)
    validate_scaled = pd.DataFrame(validate_scaled).rename(columns=zipped)
    test_scaled = pd.DataFrame(test_scaled).rename(columns=zipped)
    
    #Return the three scaled DataFrames
    return train_scaled, validate_scaled, test_scaled