from env import get_db_url
import numpy as np
import pandas as pd
import os
import acquire
import prepare
import wrangle
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import seaborn as sns



def plot_variable_pairs(df, sample_size=10000):
    
    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=823)
    else:
        df_sample = df

    
    for col in df.columns:
        for col2 in df.columns:
            if col == col2:
                continue
            else:
                sns.lmplot(x=col, y=col2, data=df_sample, line_kws={'color': 'red'})
                plt.show()


            
def plot_categorical_and_continuous_vars(df, sample_size=10000):# lists of discrete and continuous columns:
    wrangle.wrangle_zillow_predictions_without_drop()
    
    #Plots catetorical and continuous variables with the non-dummified zillo_predictions as they are better for inter county analysis      
    discrete_list = ['fips'] #took out fips
    continuous_list = ['bedrooms', 'bathrooms', 'sqft', 'tax_value', 'year_built', 'tax_amount']


    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=823)
    else:
        df_sample = df
    
    for discrete_col in discrete_list:
        for continuous_col in continuous_list:
            
            plt.figure(figsize=(11,6))
            
            plt.subplot(131)
            sns.boxplot(x=discrete_col, y=continuous_col, data=df_sample)
            plt.subplot(132)
            sns.violinplot(x=discrete_col, y=continuous_col, data=df_sample)
            plt.subplot(133)
            sns.barplot(x=discrete_col, y=continuous_col, data=df_sample)


#Visualizes the correlations of the columns to all the other columns
def correlation_heatmap(train):
    house_corr = train.corr()

    # Pass my correlation matrix to Seaborn's heatmap.

    kwargs = {'alpha':.9,'linewidth':3, 'linestyle':'-', 
          'linecolor':'k','rasterized':False, 'edgecolor':'w', 
          'capstyle':'projecting',}

    plt.figure(figsize=(8,6))
    sns.heatmap(house_corr, cmap='Blues', annot=True, mask= np.triu(house_corr), **kwargs)
    plt.ylim(0, 9)

    plt.show()



#Function to see what SelectKBest chooses as our n best features
def select_kbest(X, y, n=2):
    f_selector = SelectKBest(f_regression, k=n)
    
    f_selector.fit(X, y)
    
    X_reduced = f_selector.transform(X)
    
    f_support = f_selector.get_support()
    
    f_feature = X.loc[:, f_support].columns.tolist()
    
    print(str(len(f_feature)), 'selected features')
    print(f_feature)


#Function to see what RFE chooses as our n best features

def rfe(X, y, n=2):
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=n)
    
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X, y) 

    #fit
    lm.fit(X_rfe, y)
    
    mask = rfe.support_
    
    rfe_features = X.loc[:, mask].columns.tolist()

    print(str(len(rfe_features)), 'selected features')
    print(rfe_features)



#Split the mvp zillow_predictions
def mvp_split(train, validate, test):
    train_mvp = train.drop(columns=['year_built', 'tax_amount']) #took out fips
    validate_mvp = validate.drop(columns=['year_built', 'tax_amount'])
    test_mvp = test.drop(columns=['year_built', 'tax_amount'])

    X_train_mvp = train_mvp.drop(columns=['tax_value'])
    y_train_mvp = train_mvp['tax_value']

    X_validate_mvp = validate_mvp.drop(columns=['tax_value'])
    y_validate_mvp = validate_mvp['tax_value']

    X_test_mvp = test_mvp.drop(columns=['tax_value'])
    y_test_mvp = test_mvp['tax_value']

    return X_train_mvp, y_train_mvp, X_validate_mvp, y_validate_mvp, X_test_mvp, y_test_mvp


#Turns mvp splits into their dataframes and prints out RMSE
def baseline_rmse(y_train_mvp, y_validate_mvp, y_test_mvp):
    # turn series into dataframes to append new columns with predicted values
    y_train_mvp = pd.DataFrame(y_train_mvp)
    y_validate_mvp = pd.DataFrame(y_validate_mvp)
    y_test_mvp = pd.DataFrame(y_test_mvp)

    # 1. Predict based on mean
    tax_value_pred_mean_mvp = y_train_mvp['tax_value'].mean()
    y_train_mvp['tax_value_pred_mean'] = tax_value_pred_mean_mvp
    y_validate_mvp['tax_value_pred_mean'] = tax_value_pred_mean_mvp

    # 2. Do same for median
    tax_value_pred_median_mvp = y_train_mvp['tax_value'].median()
    y_train_mvp['tax_value_pred_median'] = tax_value_pred_median_mvp
    y_validate_mvp['tax_value_pred_median'] = tax_value_pred_median_mvp

    # 3.  RMSE of tax_value_pred_mean
    rmse_train_mvp_mean = mean_squared_error(y_train_mvp.tax_value, y_train_mvp.tax_value_pred_mean) ** (1/2)
    rmse_validate_mvp_mean = mean_squared_error(y_validate_mvp.tax_value, y_validate_mvp.tax_value_pred_mean) ** (1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train_mvp_mean, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_validate_mvp_mean, 2)) 

    # 4.  RMSE of tax_value_pred_median
    rmse_train_mvp_median = mean_squared_error(y_train_mvp.tax_value, y_train_mvp.tax_value_pred_median) ** (1/2)
    rmse_validate_mvp_median = mean_squared_error(y_validate_mvp.tax_value, y_validate_mvp.tax_value_pred_median) ** (1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train_mvp_median, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_validate_mvp_median, 2))
    return y_train_mvp, y_validate_mvp, y_test_mvp, rmse_train_mvp_median, rmse_validate_mvp_median




#Function for the Linear Regression
def linear_regression(X_train_mvp, y_train_mvp, X_validate_mvp, y_validate_mvp):
#. Create the model object
    lm_mvp = LinearRegression()

    #. Fit to training and specify column in y_train since it is now a series
    lm_mvp.fit(X_train_mvp, y_train_mvp.tax_value)

    # predict
    y_train_mvp['tax_value_pred_lm'] = lm_mvp.predict(X_train_mvp)

    # RMSE
    rmse_train_mvp_lm = mean_squared_error(y_train_mvp.tax_value, y_train_mvp.tax_value_pred_lm) ** (1/2)

    # predict validate
    y_validate_mvp['tax_value_pred_lm'] = lm_mvp.predict(X_validate_mvp)

    #Validate RMSE 
    rmse_validate_mvp_lm = mean_squared_error(y_validate_mvp.tax_value, y_validate_mvp.tax_value_pred_lm) ** (1/2)

    print('RMSE for OLS using LinearRegression\nTraining/In-Sample: ', rmse_train_mvp_lm,
        '\nValidation/Out-of-Sample: ', rmse_validate_mvp_lm)

    return y_train_mvp, y_validate_mvp, rmse_train_mvp_lm, rmse_validate_mvp_lm

#Function to call lasso_lars
def lasso_lars(X_train_mvp, y_train_mvp, X_validate_mvp, y_validate_mvp):
    lars_mvp = LassoLars(alpha=10)

    #. Fit to training and specify column in y_train since it is now a series
    lars_mvp.fit(X_train_mvp, y_train_mvp.tax_value)

    # predict
    y_train_mvp['tax_value_pred_lars'] = lars_mvp.predict(X_train_mvp)

    # RMSE
    rmse_train_mvp_lars = mean_squared_error(y_train_mvp.tax_value, y_train_mvp.tax_value_pred_lars) ** (1/2)

    # predict validate
    y_validate_mvp['tax_value_pred_lars'] = lars_mvp.predict(X_validate_mvp)

    #Validate RMSE 
    rmse_validate_mvp_lars = mean_squared_error(y_validate_mvp.tax_value, y_validate_mvp.tax_value_pred_lars) ** (1/2)

    print('RMSE  using LassoLars\nTraining/In-Sample: ', rmse_train_mvp_lars,
        '\nValidation/Out-of-Sample: ', rmse_validate_mvp_lars,
        '\With alpha= 10')
    
    return y_train_mvp, y_validate_mvp, rmse_train_mvp_lars, rmse_validate_mvp_lars


#Function for the quadratic regression

def quadratic_regression(X_train_mvp, y_train_mvp, X_validate_mvp, y_validate_mvp, X_test_mvp, y_test_mvp):
    # make the polynomial features to get a new set of features
    pf_mvp = PolynomialFeatures(degree=2)

    # fit and transform X_train
    X_train_degree2_mvp = pf_mvp.fit_transform(X_train_mvp)

    # transform X_validate & X_test
    X_validate_degree2_mvp = pf_mvp.transform(X_validate_mvp)
    X_test_degree2_mvp = pf_mvp.transform(X_test_mvp)

    # create the model object
    lm2_mvp = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2_mvp.fit(X_train_degree2_mvp, y_train_mvp.tax_value)

    # predict train
    y_train_mvp['tax_value_pred_poly'] = lm2_mvp.predict(X_train_degree2_mvp)

    # evaluate: rmse
    rmse_train_mvp_quad = mean_squared_error(y_train_mvp.tax_value, y_train_mvp.tax_value_pred_poly)**(1/2)

    # predict validate
    y_validate_mvp['tax_value_pred_poly'] = lm2_mvp.predict(X_validate_degree2_mvp)


    # evaluate: rmse
    rmse_validate_mvp_quad = mean_squared_error(y_validate_mvp.tax_value, y_validate_mvp.tax_value_pred_poly)**(1/2)

    print("RMSE for Polynomial Model, degrees=", 2, "\nTraining/In-Sample: ", rmse_train_mvp_quad, 
        "\nValidation/Out-of-Sample: ", rmse_validate_mvp_quad)
    
    # predict test
    y_test_mvp['tax_value_pred_poly'] = lm2_mvp.predict(X_test_degree2_mvp)

    # evaluate: rmse
    rmse_test_mvp_quad = mean_squared_error(y_test_mvp.tax_value, y_test_mvp.tax_value_pred_poly)**(1/2)

    return y_train_mvp, y_validate_mvp, rmse_train_mvp_quad, rmse_validate_mvp_quad, rmse_test_mvp_quad

#Scale Function I used but took out of my final product as scaling had no effect or a negative effect depending on which test I used.
def scale_zillow(df):
    # Split the Data
    train_validate, test = train_test_split(df, test_size = .2, random_state=823)
    train, validate = train_test_split(train_validate, test_size= .25, random_state=823)
    
    X_train = train.drop(columns=['year_built', 'tax_amount', 'tax_value'])#took out fips
    y_train_scaled = pd.DataFrame(train['tax_value'])

    X_validate = validate.drop(columns=['year_built', 'tax_amount', 'tax_value'])
    y_validate_scaled = pd.DataFrame(validate['tax_value'])

    X_test = test.drop(columns=['year_built', 'tax_amount', 'tax_value'])
    y_test_scaled = pd.DataFrame(test['tax_value'])
    
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
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train_scaled, y_validate_scaled, y_test_scaled
