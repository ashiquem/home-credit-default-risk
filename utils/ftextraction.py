import pandas as pd

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


