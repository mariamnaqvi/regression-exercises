import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
# use get_db_url function to connect to the codeup db
from env import get_db_url
# import to use in the split function
from sklearn.model_selection import train_test_split
# imports for feature engineering
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

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

def wrangle_data(df):
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

def Min_Max_scaler(X_train, X_validate, X_test):
    '''
    Takes in 3 pandas dataframes of X_train, X_validate and X_test. Then returns the 
    scaler object as well as the transformed dfs
    
    This function assumes the independent variables being fed in as arguments
    are all continuous features
    '''
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index,
                                 columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index=X_validate.index,
                                 columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index,
                                 columns=X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled
    
def visualize_scaled_data(scaler, scaler_name, feature):
    scaled = scaler.fit_transform(train[[feature]])
    fig = plt.figure(figsize = (12,6))

    gs = plt.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.scatter(train[[feature]], scaled)
    ax1.set(xlabel = feature, ylabel = 'Scaled_' + feature, title = scaler_name)

    ax2.hist(train[[feature]])
    ax2.set(title = 'Original')

    ax3.hist(scaled)
    ax3.set(title = 'Scaled')
    plt.tight_layout()


def split_data(df):
    '''
    This function takes in a pandas dataframe, splits it into train, test and split dataframes and returns the three split datasets.
    '''
    train, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train, train_size=0.7, random_state=123)
    return train, validate, test


def select_kbest(X, y, k):
    '''
    This function takes in the predictors (X), the target variable (y) and the number of features to select (k) and
    returns the names of the top k selected features based on the SelectKBest class.
    '''
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X, y)
    mask = f_selector.get_support()    
    f_feature = X.columns[mask]
    return f_feature


def rfe(X, y, n):
    '''
    This function takes in the predictors (X), the target variable (y) and the number of features to select (n) and
    returns the names of the top k selected features based on the Recursive Feature Engineering class.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, n)
    rfe.fit(X, y)
    feat_selected = X.columns[rfe.support_]
    return feat_selected

