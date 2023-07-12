import numpy as np
import pandas as pd
import os
from env import get_db_url


def get_zillow_data():
    filename = 'zillow.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    else:
        # read the SQL query into a dataframe
        zillow_db = pd.read_sql('''
                                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                                FROM properties_2017
                                WHERE propertylandusetypeid = 261 -- Single Family Residence
                                ''', get_db_url('zillow'))
        
        # Write that dataframe to disk for later.  Called "caching" the data for later.
        zillow_db.to_csv(filename)
        
        return zillow_db
    
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