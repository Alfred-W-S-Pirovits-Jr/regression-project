import numpy as np
import pandas as pd
import os
from env import get_db_url
import acquire
from sklearn.model_selection import train_test_split

# Split the data
def split_data(df, stratify_col=None):
    if stratify_col == None:
        train_validate, test = train_test_split(df, test_size = .2, random_state=823)
        train, validate = train_test_split(train_validate, test_size=.25, random_state=823)
        return train, validate, test

    else:
        train_validate, test = train_test_split(df, test_size = .2, random_state=823, stratify=df[stratify_col])
        train, validate = train_test_split(train_validate, test_size=.25, random_state=823, stratify=train_validate[stratify_col])
        return train, validate, test

#Sequel query to get the 2017 zillow transactions
def get_zillow_predictions():
    filename = 'zillow_predictions.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    else:
        # read the SQL query into a dataframe
        zillow_db = pd.read_sql('''
                                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
                                FROM properties_2017
                                INNER JOIN predictions_2017 ON (properties_2017.parcelid = predictions_2017.parcelid)
                                WHERE propertylandusetypeid = 261; -- Single Family Residence
                                ''', get_db_url('zillow'))
        
        # Write that dataframe to disk for later.  Called "caching" the data for later.
        zillow_db.to_csv(filename)
        
        return zillow_db

#prepare the file by dropping na's and renaming columns as well
def wrangle_zillow_predictions():
    zillow = get_zillow_predictions()
    zillow = zillow.dropna()
    zillow = zillow.rename(columns = {'bedroomcnt': 'bedrooms',
                                 'bathroomcnt': 'bathrooms',
                                 'calculatedfinishedsquarefeet': 'sqft',
                                 'taxvaluedollarcnt': 'tax_value',
                                 'yearbuilt': 'year_built',
                                 'taxamount': 'tax_amount'})
    zillow.drop_duplicates(inplace=True)

    #Create 3 categorical columns for fips named for the counties
    zillow_dummies = pd.get_dummies(zillow['fips'])
    zillow_dummies = zillow_dummies.rename(columns={6037.0 : "Los Angeles", 6059.0 : "Orange", 6111.0 : "Ventura"})
    zillow_predictions = pd.concat([zillow, zillow_dummies], axis=1).drop(columns='fips')
    return zillow_predictions

#kept for better graphs categorical and discrete
def wrangle_zillow_predictions_without_drop():
    zillow = get_zillow_predictions()
    zillow = zillow.dropna()
    zillow = zillow.rename(columns = {'bedroomcnt': 'bedrooms',
                                 'bathroomcnt': 'bathrooms',
                                 'calculatedfinishedsquarefeet': 'sqft',
                                 'taxvaluedollarcnt': 'tax_value',
                                 'yearbuilt': 'year_built',
                                 'taxamount': 'tax_amount'})
    zillow.drop_duplicates(inplace=True)
    return zillow