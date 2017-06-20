import pandas as _pd
import numpy as np
from scipy.stats import chi2_contingency as _chi2
from scipy.stats import fisher_exact
import matplotlib as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact
from sklearn import preprocessing
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt

# CATEGORICAL PREDICTORS and CATEGORICAL OUTCOMES
def contingency_table(predictor, outcome, dataframe):
    '''
    contingency_table(predictor, outcome, dataframe)
    '''
    T = _pd.crosstab(dataframe[predictor], dataframe[outcome]).astype(float);
    T = T.loc[['Yes','No']]
    T = T[['Yes','No']]
    return T

def relative_risk(T):
    '''
    source: https://www.medcalc.org/calc/relative_risk.php
    RR, lb, ub  = relative_risk(T)
    Estimate the relavite risk (RR), its lower 95%CI (lb), and its upper 95%CI(ub)
    '''
    a = T[0,0]
    b = T[0,1]
    c = T[1,0]
    d = T[1,1]
    SE = np.sqrt( 1/a + 1/c - 1/(a+b) - 1/(c+d) )

    p_of_first_row = a / (a+b)
    p_of_seconds_row = c / (c+d)
    RR = p_of_first_row/p_of_seconds_row

    SE = np.sqrt( 1/a + 1/c - (1/(a+b)) - (1/(c+d)) )
    CI_95pct_lb = np.exp(np.log(RR) - 1.96 * (SE))
    CI_95pct_ub = np.exp(np.log(RR) + 1.96 * (SE))

    return np.round(RR,4), np.round(CI_95pct_lb,4), np.round(CI_95pct_ub,4)

def categorical_data(outcome, categorical_predictors, df):
    '''
    --------
    Syntaxis
    categorical_data(outcome, categorical_predictors, df)
    --------
    Inputs:
    outcome = string with the categorical outcome to be studied
    predictors = list of strings with the categorical predictors
    df = Pandas Data Frame with the data
    -------
    Returns:
    Pandas df with the following columns for each predictor
    p-value : p-value of chi-squared test
    Relative_Risk: Relative Risk (RR) of first row vs second row
    RR_lb: Lower bound of the 95% C.I for the RR
    RR_ub: Upper bound of the 95% C.I for the RR
    '''
    #categorical_predictors = df.columns[2:7]
    num_pred = len(categorical_predictors)
    df2 = _pd.DataFrame(np.random.randn(num_pred, 4))
    df2 = df2.set_index([categorical_predictors])
    df2.columns = ['p-value', 'Relative_Risk','RR_lb','RR_ub']

    for idx, var in enumerate(categorical_predictors):
        T = contingency_table(var, outcome, df)
        _, p , _, _= _chi2( T.values )
        RR, lb, ub = relative_risk(T.values)

        df2.iloc[idx,0] = p;
        df2.iloc[idx,1] = RR;
        df2.iloc[idx,2] = lb;
        df2.iloc[idx,3] = ub;
    return df2

# Continous PREDICTORS and CATEGORICAL OUTCOMES


def linear_models(df, outcome, predictors, print_results = 1):
    # create new dataframe with predictors
    df2 = _pd.DataFrame()
    df2[predictors] = df[predictors]

    # ad outcome to dataframe with predictors encoded as floating
    df2[outcome]   = preprocessing.LabelEncoder().fit_transform( df[outcome].values )

    # create formula from strings
    formula = outcome + "~" + "+".join(predictors )

    # perform ANOVA
    SPY_lm = ols(formula, data = df2 ).fit()
    anova = sm.stats.anova_lm(SPY_lm, typ=2) # Type 2 ANOVA DataFrame
    if print_results == 1:
        print(15*'---')
        print(anova)
        print(15*'---')
    return anova, SPY_lm

def anova_MRI(outcome, df):
    mri_predictors= ['MRI_LD_Baseline','MRI_LD_1_3dAC', 'MRI_LD_Int_Reg', 'MRI_LD_PreSurg']
    results = results = _pd.DataFrame(np.random.random(size=(len(mri_predictors),1)),
                                          index=mri_predictors, columns=['p-value'])
    for idx, pred in enumerate(mri_predictors):
        p = list();     p.append(pred)
        nova_table, _ = linear_models(df, outcome, p, print_results=0);
        results.ix[idx] = nova_table['PR(>F)'].values[0]

    f, (ax1, ax2) = plt.subplots(2,2, figsize=(10,10))
    sns.boxplot(x= outcome, y=mri_predictors[0], data=df, palette="Set3", ax=ax1[0]).set_title('p-value = '+ str(results.values[0]));
    sns.boxplot(x= outcome, y=mri_predictors[1], data=df, palette="Set3", ax=ax2[0]).set_title('p-value = '+ str(results.values[1]));
    sns.boxplot(x= outcome, y=mri_predictors[2], data=df, palette="Set3", ax=ax1[1]).set_title('p-value = '+ str(results.values[2]));
    sns.boxplot(x= outcome, y=mri_predictors[3], data=df, palette="Set3", ax=ax2[1]).set_title('p-value = '+ str(results.values[3]));
    plt.show()
    return results

def effect_size(df,predictors, outcome):
    all_ =  predictors + [outcome]
    legend = 'Predictor of ' + outcome
    mean_ = df[all_].groupby( outcome ).mean().values
    std_ =  np.std( df[predictors].values.flatten() )
    delta = mean_[0,:] - mean_[1,:]
    effect_size = _pd.DataFrame(delta/std_)

    effect_size[legend] = predictors
    effect_size =    effect_size.set_index(legend)
    effect_size.columns = ['Effect Size']
    return effect_size
