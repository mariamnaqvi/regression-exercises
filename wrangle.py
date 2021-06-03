import pandas as pd
import numpy as np
import os
# use get_db_url function to connect to the codeup db
from env import get_db_url

def get_telco_data(cached=False):
    '''
    This function returns the telco churn database as a pandas dataframe. 
    If the data is cached or the file exists in the directory, the function will read the data into a df and return it. 
    Otherwise, the function will read the database into a dataframe, cache it as a csv file
    and return the dataframe.
    '''
    # If the cached parameter is false, or the csv file is not on disk, read from the database into a dataframe
    if cached == False or os.path.isfile('telco_df.csv') == False:
        sql_query = '''
        SELECT customer_id, monthly_charges, tenure, total_charges
        FROM customers 
        WHERE contract_type_id = 3;
        '''
        telco_df = pd.read_sql(sql_query, get_db_url('telco_churn'))
        #also cache the data we read from the db, to a file on disk
        telco_df.to_csv('telco_df.csv')
    else:
        # either the cached parameter was true, or a file exists on disk. Read that into a df instead of going to the database
        telco_df = pd.read_csv('telco_df.csv', index_col=0)
    # return our dataframe regardless of its origin
    return telco_df

def wrangle_telco(df):
    '''
    This function takes in a pandas dataframe and checks it for duplicates. It also converts the empty string values in 
    total charges to nan values and replaces them with monthly charges values. It then prints the .info() to verify that
    the changes have been made and returns the df.
    '''
    # check for duplicates 
    num_dups = df.duplicated().sum()
    # if we found duplicate rows, we will remove them, log accordingly and proceed
    if num_dups > 0:
        print(f'There are {num_dups} duplicate rows in your dataset - these will be dropped.')

        print ('----------------')
        # remove duplicates if found
        df = df.drop_duplicates()
    else:
        # otherwise, we log that there are no dupes, and proceed with our process
        print(f'There are no duplicate rows in your dataset.')
    # replace empty strings in total_charges with nan values     
    df.total_charges = df.total_charges.replace(' ', np.nan)
    # replace nan values with monthly charges
    df.total_charges = df.total_charges.fillna(value=df.monthly_charges)
    # convert total charges to float
    df.total_charges = df.total_charges.astype(float)
    # print the .info so we can verify that all changes we wanted have been made
    print(df.info())
    return df

