import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from numpy import round
import matplotlib.pyplot as plt

# load

def create_table(DF):
    df = DF[['race','call']].copy()
    Con_Table = pd.crosstab(df.race,df.call)
    Con_Table.columns = ['No','Yes']
    Con_Table.index = ['Black','White']
    return Con_Table, df

def my_chi2(X):
    '''
    chi2, p, _, expected_distribution = scipy.stats.chi2_contingency(X,correction=True)
    '''
    chi2, p, _, expected_distribution = scipy.stats.chi2_contingency(X,correction=True)
    print('The p-value is : '+str(p))
    print('The test statistic is : '+str(chi2))
    print('The expected distribution is :')
    print(round(expected_distribution))
    
def prepare_for_classification(DF):
    '''
    X = yearsexp, 'sex', education, 'race', manager
    Y = call 
    '''
    df = DF[['yearsexp','sex','education','race','manager']].copy()
    df['sex'][df['sex']=='f']=1.0
    df['sex'][df['sex']=='m']=0.0
    df['race'][df['race']=='w']=0.0
    df['race'][df['race']=='b']=1.0
    X = df.values
    y = DF.call.values
    df['Call'] = DF.call
    return X,y
    
def LogRe(X,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    clf = LogisticRegression()
    clf.fit(X, y)
    preds = clf.predict_proba(X)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, preds)
    auc = metrics.auc(fpr,tpr)
    plt.figure(1)
    plt.plot(fpr,tpr,'o-');
    plt.title("AUC =" + str(auc))
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.show()
    return fpr, tpr, auc
    

    
