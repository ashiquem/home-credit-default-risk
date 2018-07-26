import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc
gc.enable()

class modeltraining:
    """Methods for training and testing different 
       machine learning models
     """

    def train_predict_lgbm(self,train,test,folds,iterations,exclude=[]):
        """Function to train a Gradient Boosted Model,
           based on the lightgbm library.

           Parameters:
           ----------
           train(pandas dataframe): training dataset
           test(pandas dataframe): testing dataset
           folds(integer): number of folds to use for k-fold cross validation
           iterations(integer): number of iterations used for early stopping criteria 
           exclude(list string): column names to exclude from dataset

           Returns:
           --------
           submission(pandas dataframe): predictions on the test set
           cv_score(pandas dataframe): cross validation scores
           feature_importances(pandas dataframe): feature importances for the model

        """
        #PREPROCESSING DATA
        
        # save test IDs for final submission dataframe
        test_IDs = test['SK_ID_CURR']
        # save target labels
        labels = train['TARGET']
        # drop unnecessary columns
        train = train.drop(columns=['SK_ID_CURR','TARGET']+exclude)
        test = test.drop(columns=['SK_ID_CURR']+exclude)

        # encode categorical variables
        train = pd.get_dummies(train)
        test = pd.get_dummies(test)
        
        # aligining dataframes
        train,test = train.align(test,join='inner',axis=1)
        
        # for storing feature importances
        features = list(train.columns)
        ft_importances = np.zeros(len(features))
        
        # converting to numpy array for lgbm consumptions
        train = np.array(train)
        test = np.array(test)
        
        #DATA STRUCTURES TO STORE PREDICTIONS AND METRICS
        
        #store cv predictions
        oof_predictions = np.zeros(train.shape[0])
        #store predictions on test dataset
        test_preds = np.zeros(test.shape[0])
        #store ROC score for cv predictions
        cv_roc_train = []
        cv_roc_valid = []
        cv_score = pd.DataFrame()
        
        #SPLITTING AND TRAINING 
        kfold = KFold(n_splits=folds,shuffle=True,random_state=40)
        
        for train_i,valid_i in kfold.split(train):
            xtrain,ytrain = train[train_i],labels[train_i]
            xvalid,yvalid = train[valid_i],labels[valid_i]
            
            # creating the classifier 
            clf = lgbm.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                    class_weight = 'balanced', learning_rate = 0.05, 
                                    reg_alpha = 0.1, reg_lambda = 0.1, 
                                    subsample = 0.8, n_jobs = -1, random_state = 50)
            
            # fitting on the training set, early stopping using validation set
            clf.fit(xtrain,ytrain,eval_set=[(xtrain, ytrain), (xvalid, yvalid)],eval_metric ='auc',
                verbose= 200, early_stopping_rounds= iterations,eval_names = ['train','valid'])
            
            # recording best iteration
            best_iter = clf.best_iteration_
            
            # storing out of fold predictions:
            oof_predictions[valid_i] = clf.predict_proba(xvalid,num_iteration=best_iter)[:, 1]
            
            # storing training and validation scores
            cv_roc_train.append(clf.best_score_['train'])
            cv_roc_valid.append(clf.best_score_['valid'])
            
            # storing test set predictions:
            test_preds += clf.predict_proba(test,num_iteration=best_iter)[:, 1] /kfold.n_splits
            
            # storing feature importances
            
            ft_importances += clf.feature_importances_ /kfold.n_splits
            
            # freeing up memory
            del xtrain,ytrain,xvalid,yvalid,clf
            gc.collect()
            
        #SCORES
        
        feature_importances = pd.DataFrame({'features':features,'importance':ft_importances})
        
        cv_score['Fold'] = np.arange(kfold.n_splits)
        cv_score['Training Score'] = cv_roc_train
        cv_score['Validation Score'] = cv_roc_valid
        overall = roc_auc_score(labels,oof_predictions)
        print('Overall CV Score: %.2f' %overall)
        # submission dataframe:
        submission = pd.DataFrame({'SK_ID_CURR': test_IDs, 'TARGET': test_preds})
        
        return submission, cv_score,feature_importances