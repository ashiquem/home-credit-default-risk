import pandas as pd
import numpy as np
import gc
import time
gc.enable()

class Extractor:
    """Functions for computing feature statistics on pandas dataframes"""

    def rename_columns(self,groupby,df,df_name):
        """Function to rename pandas dataframe with aggregated statistics
           
           Parameters:
           -----------
           groupby(string): column name used to group
           df(pandas dataframe): dataframe to be renames
           df_name(string): dataset name

           Returns:

           df(pandas datafram): dataframe with renamed columns 
        
        """

        col_names = [groupby]

        for col in df.columns.levels[0]:
            if col != groupby:
                for stat in df[col].columns:
                    col_names.append('%s_%s_%s'%(df_name,col,stat))

        df.columns = col_names
        return df

    def encode_categorical(self,df,nan_col=True):
        """Function to encode categorical features
           Uses label encoding for features with <=2 unique labels
           Uses one-hot-encoding for the rest.

           Parameters:
           -----------
           df(pandas dataframe): dataset to encode
           nan_col(bool): true to include nans as columns
           Returns:
           --------
           cat_cols(list of string): column names of encoded features
           df(pandas dataframe): encoded dataset
        """
        all_cols = list(df.columns)
        cat_cols = []
        #binary encoding:
        for col in df.select_dtypes('object'):
            if(len(df[col].unique())<=2):
                df[col],uniques = pd.factorize(df[col])
                cat_cols.append(col)
        
        #one-hot-encoding:
        
        df = pd.get_dummies(df,columns=list(df.select_dtypes('object')),dummy_na=nan_col)
        for col in df.columns:
            if col not in all_cols:
                cat_cols.append(col)
                
        return cat_cols,df

    

    def aggregate_stats(self,gb,df,df_name=""):
        """Function to compute aggregate statistics
           
           Parameters:
           -----------
           gb(string): feature to aggregate by
           df(pandas dataframe): dataset to perform operation on

           Returns:
           --------
           df(pandas dataframe): dataset with aggregated stats
        """

        #numerical columns:
        num_cols = list(df.select_dtypes('number').columns)
        #define numerical aggregations:
        num_aggs = {}
        for c in num_cols:
            if c != gb:
                num_aggs[c] = ['mean','max','min','count','sum']
        
        #encode categorical features:
        cat_cols,df = self.encode_categorical(df)

        #define categorical aggregations:
        cat_aggs = {}
        for c in cat_cols: cat_aggs[c]=['count','mean']
        
        #compute grouped aggregate stats:
        df = df.groupby(gb).agg({**num_aggs,**cat_aggs}).reset_index()
        
        #renamedf
        df = self.rename_columns(gb,df,df_name)

        return df


    def apptrain_test_data(self,df,test,path='',fe=True):
        allapps = df.append(test).reset_index(drop=True)

        #null days employed anomaly
        allapps['DAYS_EMPLOYED'].replace(365243,np.nan,inplace=True)
        if fe == True:
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

        #encode categorical features
        catcols, allapps = self.encode_categorical(allapps)

        del test
        gc.collect()

        if path != '':
            print(f'Saving dataframe to {path}')
            allapps.to_csv(f'{path}',index=False)

        return allapps

    def previous_applications(self,pa,path='',fe=True):

        #null anomalies
        pa['DAYS_FIRST_DRAWING'].replace(365243,np.nan,inplace=True)
        pa['DAYS_FIRST_DUE'].replace(365243,np.nan,inplace=True)
        pa['DAYS_LAST_DUE_1ST_VERSION'].replace(365243,np.nan,inplace=True)
        pa['DAYS_LAST_DUE'].replace(365243,np.nan,inplace=True)
        pa['DAYS_TERMINATION'].replace(365243,np.nan,inplace=True)

        if fe == True:
            #create new features
            pa['CREDIT_TO_APP'] = pa['AMT_CREDIT']/pa['AMT_APPLICATION']
            pa['PA_CREDIT_ANNUITY'] = pa['AMT_CREDIT']/pa['AMT_ANNUITY']

        #aggregate stats:
        pa.drop(columns='SK_ID_PREV',inplace=True)
        pa = self.aggregate_stats('SK_ID_CURR',pa,'PA')

        if path != '':
            print(f'Saving dataframe to {path}')
            pa.to_csv(f'{path}',index=False)


        return pa

    def bureau_and_balance(self,buro,bb,path=''):
        #aggregate statistics on bureau balance:
        bb_agg = self.aggregate_stats('SK_ID_BUREAU',bb,'BB')
        del bb
        gc.collect()

        #join to bureau data:
        buro = pd.merge(buro,bb_agg,on='SK_ID_BUREAU',how='left')
        
        #drop bureau balance id:
        buro.drop(columns='SK_ID_BUREAU',inplace=True)

        #aggregate statistics on bureau data:
        buro_agg = self.aggregate_stats('SK_ID_CURR',buro,'BURO')

        del buro
        gc.collect()

        if path != '':
            print(f'Saving dataframe to {path}')
            buro_agg.to_csv(f'{path}',index=False)

        return buro_agg

    def installment_payments(self,df,pa,path=''):
        #aggregate stats by previous applications:
        df.drop(columns='SK_ID_CURR',inplace=True)
        ipay_prv = self.aggregate_stats('SK_ID_PREV',df,'ipay')
        
        del df
        gc.collect()

        #aggregate stats by current load ID
        ipay_prv = pd.merge(pa,ipay_prv,on='SK_ID_PREV',how='left')
        ipay_prv.drop(columns='SK_ID_PREV',inplace=True)
        ipay_prv = self.aggregate_stats('SK_ID_CURR',ipay_prv,'client')

        del pa
        gc.collect()

        if path != '':
            print(f'Saving dataframe to {path}')
            ipay_prv.to_csv(f'{path}',index=False)

        return ipay_prv

    def cc_balance(self,df,pa,path=''):
        #aggregate stats by previous applications
        df.drop(columns='SK_ID_CURR',inplace=True)
        ccb = self.aggregate_stats('SK_ID_PREV',df,'ccb')
        
        del df
        gc.collect()

        #aggregate by current loan ID
        ccb = pd.merge(pa,ccb,on='SK_ID_PREV',how='left')
        ccb.drop(columns='SK_ID_PREV',inplace=True)
        ccb = self.aggregate_stats('SK_ID_CURR',ccb,'client')

        del pa
        gc.collect()

        if path != '':
            print(f'Saving dataframe to {path}')
            ccb.to_csv(f'{path}',index=False)


        return ccb

    def pos_cash_balance(self,df,pa,path=''):
        #aggregate by previous applications:
        df.drop(columns='SK_ID_CURR',inplace=True)
        pos = self.aggregate_stats('SK_ID_PREV',df,'pos')

        del df
        gc.collect()

        #aggregate by current load ID
        pos = pd.merge(pa,pos,on='SK_ID_PREV',how='left')
        pos.drop(columns='SK_ID_PREV',inplace=True)
        pos = self.aggregate_stats('SK_ID_CURR',pos,'client')

        del pa
        gc.collect()

        if path != '':
            print(f'Saving dataframe to {path}')
            pos.to_csv(f'{path}',index=False)


        return pos

    def drop_missing_values(self,df,threshold):
        missing = df.isnull().sum()
        count = missing[missing !=0]
        percent = (count/len(df)*100).round(1)
        missing = pd.concat([count,percent],axis=1,keys=['Count','Percent'])

        to_drop = list(missing[missing['Percent']>threshold].index)
        print('Dropping %d columns....'%len(to_drop))
        df.drop(columns=to_drop,inplace=True)
        
        return df

    def process_datasets(self,apptrain,apptest,buro,bb,pa,ipay,ccb,pos,fe=True,path=''):
        """Process all datasets
           
           Parameters:
           -----------
           apptrain,apptest,buro,bb,pa,ipay,ccb,pos(pandas dataframe): datasets
           fe(boolean): True procesess with engineered features
           path(string): path to save processed file
        
        """
        start = time.time()
        print('Aggregating data sets....')
        #encoding and aggregating datasets
        appdata = self.apptrain_test_data(apptrain,apptest,fe=fe)
        buro_data = self.bureau_and_balance(buro,bb)
        paIds = pa[['SK_ID_PREV','SK_ID_CURR']]
        pa = self.previous_applications(pa,fe=fe)
        ipay = self.installment_payments(ipay,paIds)
        ccb = self.cc_balance(ccb,paIds)
        pos = self.pos_cash_balance(pos,paIds)
        end = time.time()
        print('Aggregation done in %.2f minutes '%((end-start)/60))
        del apptrain, apptest, buro, bb
        gc.collect()

        print('Joining datasets.... ')
        start = time.time()
        #joining datasets
        appdata = pd.merge(appdata,buro_data,on='SK_ID_CURR',how='left')
        appdata = pd.merge(appdata,pa,on='SK_ID_CURR',how='left')
        appdata = pd.merge(appdata,ipay,on='SK_ID_CURR',how='left')
        appdata = pd.merge(appdata,ccb,on='SK_ID_CURR',how='left')
        appdata = pd.merge(appdata,pos,on='SK_ID_CURR',how='left')
        end = time.time()
        print('Merge done in %.2f minutes '%((end-start)/60))
        if path != '':
            appdata.to_csv(path,index=False)

        return appdata
