import pandas as pd
import numpy as np
import gc
gc.enable()

class Extractor:
    """Functions for computing feature statistics on pandas dataframes"""

    def numerical_feature_stats(self,df,gb,df_name,exclude=[]):
        """Function to calculative cumulative statistics for numerical features
        
           Parameters:
           ----------
           df(pandas dataframe): features dataframe
           gb(string): feature to groupby for cumulative stats
           df_name(string): dataset name for renaming columns
           exclude(list of string): feature names to exclude from calculations

           Returns:
           --------
           numerical_stats(pandas dataframe): dataframe containing aggregated
                                        cumulative statistics of numerical features 
        """

        for col in df:
            if col in exclude:
                df = df.drop(columns=col)

        numerical_stats = df.select_dtypes('number')

        numerical_stats = numerical_stats.groupby(gb,as_index=False).agg(['count','mean','max','min','sum']).reset_index()
        columns = [gb]

        for feature in numerical_stats.columns.levels[0]:
            if feature != gb:
                for stat in numerical_stats.columns.levels[1][:-1]:
                    columns.append('%s_%s_%s'%(df_name,feature,stat))
            
        numerical_stats.columns = columns

        return numerical_stats


    def categorical_stats(self,df,gb,df_name,exclude=[]):
        """Extracts counts and normalized counts of categorical features
        
           Parameters:
           ----------
           df(pandas dataframe): features dataframe
           gb(string): feature to groupby for cumulative stats - ID expected to be of type number
           df_name(string): dataset name for renaming columns
           exclude(list of string): feature names to exclude from calculations

           Returns:
           --------
           categorical_stats(pandas dataframe): dataframe containing aggregated
                                                cumulative statistics of categorical features       
        """
        for col in df.columns:
            if col in exclude:
                df.drop(columns=col)

        categorical_stats = pd.get_dummies(df.select_dtypes('object'))
        categorical_stats[gb] = df[gb]

        categorical_stats = categorical_stats.groupby(gb).agg(['sum','mean']).reset_index()

        columns = [gb]

        for col in categorical_stats.columns.levels[0]:
            if col != gb:
                for stat in ['count','norm_count']:
                    columns.append('%s_%s_%s' %(df_name,col,stat))

        categorical_stats.columns = columns

        return categorical_stats

    def apptrain_test_data(self,df,test):
        allapps = df.append(test).reset_index(drop=True)

        #null days employed anomaly
        allapps['DAYS_EMPLOYED'].replace(365243,np.nan,inplace=True)
        #create new features
        allapps['CREDIT_ANNUITY_RATIO'] = allapps['AMT_CREDIT']/allapps['AMT_ANNUITY']
        allapps['CREDIT_INCOME_RATIO'] = allapps['AMT_CREDIT']/allapps['AMT_INCOME_TOTAL']
        allapps['ANNUITY_INCOME_RATIO'] = allapps['AMT_ANNUITY']/allapps['AMT_INCOME_TOTAL']
        allapps['AGE_CAR_AGE_RATIO'] = allapps['DAYS_BIRTH']/allapps['OWN_CAR_AGE']
        allapps['GOODS_PRICE_INCOME_RATIO'] = allapps['AMT_GOODS_PRICE']/allapps['AMT_INCOME_TOTAL']
        allapps['INCOME_FAMILY'] = allapps['AMT_INCOME_TOTAL']/(allapps['CNT_FAM_MEMBERS'] + allapps['CNT_CHILDREN'])
        allapps['EXT_SOURCES_MEAN'] = allapps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        allapps['EXT_SOURCES_PROD'] = allapps['EXT_SOURCE_1'] * allapps['EXT_SOURCE_2'] * allapps['EXT_SOURCE_3']
        allapps['RR_MEAN'] = allapps[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].mean(axis=1)
        allapps['EMPLPYOED_TO_AGE'] = allapps['DAYS_EMPLOYED']/allapps['DAYS_BIRTH']

        del test
        gc.collect()
        print('Saving dataframe....')
        allapps.to_csv('processed/testtrain.csv',index=False)
        return allapps

    def previous_applications(self,pa):

        #null anomalies
        pa['DAYS_FIRST_DRAWING'].replace(365243,np.nan,inplace=True)
        pa['DAYS_FIRST_DUE'].replace(365243,np.nan,inplace=True)
        pa['DAYS_LAST_DUE_1ST_VERSION'].replace(365243,np.nan,inplace=True)
        pa['DAYS_LAST_DUE'].replace(365243,np.nan,inplace=True)
        pa['DAYS_TERMINATION'].replace(365243,np.nan,inplace=True)

        #creating features
        pa['CREDIT_TO_APP'] = pa['AMT_CREDIT']/pa['AMT_APPLICATION']
        pa['PA_CREDIT_ANNUITY'] = pa['AMT_CREDIT']/pa['AMT_ANNUITY']

        p_apps_numerical = self.numerical_feature_stats(pa,'SK_ID_CURR','prv_app',exclude=['SK_ID_PREV'])
        p_app_cat = self.categorical_stats(pa,'SK_ID_CURR','prv_app',exclude=['SK_ID_PREV'])
        
        p_app_cat.to_csv('processed/p_apps_cat.csv',index=False)
        p_apps_numerical.to_csv('processed/p_apps_num.csv',index=False)

        del pa
        gc.collect()

        return p_app_cat, p_apps_numerical




