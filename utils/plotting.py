import matplotlib.pyplot as plt
import seaborn as sns

class Visualizations:
    """Functions for creating plots
       
       Required packages:
       ------------------
       matplotlib
       seaborn
    """

    def plot_fi(self,fi_df):
        """create horizontal bar plots displaying feature importances
        parameters:
        ------------
        fi_df(dataframe): pandas dataframe containing the features importances
        """

        top_features = fi_df[fi_df['importance'] != 0].sort_values('importance',ascending=False).head(20)
        figure = plt.figure(figsize=(8,10))
        sns.barplot(x='importance',y='features',data=top_features)
        plt.tight_layout
        plt.title('Feature Importances')

    
    def plot_bars(self,data,features,target_label):
        """create bar plots for features
        parameters:
        ------------
        data(dataframe): pandas dataframe containing the features
        featrues(list of strings):  containing feature names
        target_label(string): target label 
        """
        width = 0.5
        rows = len(features)
        for i,feature in enumerate(features):
            defaulters = data.loc[data[target_label] == 1, feature].value_counts()
            repaid = data.loc[data[target_label]==0,feature].value_counts()
            datasets = [repaid,defaulters]
            
            for k,dataset in enumerate(datasets):
                for clas in dataset.index:
                    if clas not in datasets[~k]:
                        datasets[~k][clas] = 0

            ind1 = np.arange(len(repaid.values))*1.5
            ind2 = ind1 + 0.5
            plt.subplot(rows,1,i+1)
            plt.bar(ind1, repaid.values,width,color='blue',alpha=0.7)
            plt.bar(ind2, defaulters.values,width,color='yellow',alpha=0.9)
            plt.xticks(ind1+0.25,list(repaid.index),rotation=45)
            plt.title(f'Loan Repayment in terms of {feature}')
            plt.legend(labels=['Repayed','Defaulted'])    
    
    def plot_distributions(self,dataframe,features,target_label):
        """create KDE plots for features
        parameters:
        ------------
        dataframe: pandas dataframe containing the features
        featrues: list of strings containing feature names
        target_label: target label string
        """
        rows = math.ceil(len(features)/3)
        for i,feature in enumerate(features):
            plt.subplot(rows,3,i+1)
            sns.kdeplot(dataframe.loc[dataframe[target_label] ==0,feature],label='Repayed')
            sns.kdeplot(dataframe.loc[dataframe[target_label]==1,feature],label='Defaulted')
            plt.title(f'Distribution of {feature}')    