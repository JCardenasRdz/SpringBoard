import pandas as pd
import numpy as np
from scipy.stats import kstest, normaltest, ttest_1samp, wilcoxon, ranksums, ttest_ind
import matplotlib.pyplot as plt

def explain_test(pval,test):
    if pval > 0.05:
        print('The '+ test + ' indicates that the data is NOT normally distirbuted')
    else:
        print('The '+ test + ' indicates that the data IS normally distirbuted')

def norm_tests(xdata):
    _, p =kstest(xdata, 'norm')
    explain_test(p,'Kolmogorov-Smirnov test')
    print(40*'==')
    _, p =normaltest(xdata)
    explain_test(p,'D’Agostino and Pearson’s')

def by_gender(df):
    x = df.temperature
    females = x[(df.gender == 'F').values]
    males = x[(df.gender == 'M').values]
    return females, males

def check_gender_effect(females,males):
    print('FEMALE subjects')
    print(20*'--')
    norm_tests(females);
    print(20*' ')
    print('MALE subjects')
    print(20*' ')
    norm_tests(males);
    plt.hist(females);plt.hist(males);
    plt.legend(('Females','Males'))

def one_sample_test(xvector,mu):
    _, p_val = ttest_1samp(xvector,mu)
    print('The p-value for the t-statistic is: ' + str(p_val))
    if p_val < 0.05:
        print('We can reject the null hypothesis; The mean IS different than '+ str(mu))
    else:
        print('We cannot reject the null hypothesis; The mean is NOT different than '+ str(mu))
 
def expected_rante(observations):
    SE = np.std(observations) / np.sqrt(len(observations))
    lb = np.mean(observations) - 1.96*SE
    ub = np.mean(observations) + 1.96*SE
    print('The lower limit at the 95% conf. level is ' + str(lb))
    print('The upper limit at the 95% conf. level is ' + str(ub))
    print(10*'--')
    return lb, ub

def men_vs_women(fem,mal):
    _, p = ttest_ind(fem,mal,equal_var=False)
    print('The p-value for the t-statistic is: ' + str(p))
    if p < 0.05:
        print('We can reject the null hypothesis; The means ARE different')
    else:
        print('We cannot reject the null hypothesis; The means ARE NOT different') 
