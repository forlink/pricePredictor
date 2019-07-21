
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC

def create_features(prices):
    prices_m=(prices.high+prices.low+prices.close)/3
    n_days=5
    features=pd.concat([
                prices_m.shift(i)/prices_m.shift(i+1)-1 for i in range(0,n_days)
            ],axis=1)
    return features

def create_labels(prices,classification=True):
    prices_m=(prices.high+prices.low+prices.close)/3
    returns=pd.Series((prices_m.shift(-1)/prices_m-1),name='labels')
    if classification:
        labels=pd.qcut(returns,2,labels=False)*2-1
    else:
        labels=returns
    return labels

def create_training_df(prices,classification=True):
    features=create_features(prices)
    labels=create_labels(prices,classification=classification)

    training_df=pd.concat([features,labels],axis=1)
    training_df=training_df[training_df['labels']!=0]
    training_df=training_df.dropna()
    training_df=training_df.sample(frac=1).reset_index(drop=True)
    return training_df

def create_prediction_sample(prices):
    features=create_features(prices)
    return features.iloc[-1,:].values.reshape(1,-1)

class classifier:
    def __init__(self):
        #self.clf=KNeighborsClassifier(n_neighbors=101)
        pass

    def train(self,prices_df,verbose=False):
        clf_list=[]
        f1_list=[]
        for i in range(0,1):
            examples_df=create_training_df(prices_df,True)
            m=len(examples_df)
            n=len(examples_df.columns)
            training_df=examples_df.head(m-m//4)
            test_df=examples_df.tail(m//4)
            features=training_df.iloc[:,0:n-1]
            labels=training_df['labels']
            clf_list.append(LinearSVC(fit_intercept=False))
            clf_list[-1].fit(features.values,labels.values)
            y_pred=clf_list[-1].predict(features.values)
            y_true=labels
            if verbose:
                print('------------------------------------')
                print('Training set')
                print(classification_report(y_true, y_pred))
                print('------------------------------------')

            features_test=test_df.iloc[:,0:n-1]
            labels_test=test_df['labels']
            y_pred_test = clf_list[-1].predict(features_test.values)
            y_true_test=labels_test
            y_true_test=np.array(y_true_test)
            y_pred_test=np.array(y_pred_test)
            if verbose:
                print('------------------------------------')
                print('Test set')
                print(classification_report(y_true_test, y_pred_test))
                print('------------------------------------')
            f1_list.append(f1_score(y_true_test, y_pred_test))

        self.clf=clf_list[f1_list.index(max(f1_list))]
        return max(f1_list)

    def predict(self,prices_df,verbose=False):
        features=create_features(prices_df).iloc[-1,:]
        y_pred = self.clf.predict(features.values.reshape(1,-1))
        return y_pred
