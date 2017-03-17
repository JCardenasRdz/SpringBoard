import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from numpy import round

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
    
