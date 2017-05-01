import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
import matplotlib as plt

# load data (already cleaned)
df = pd.read_csv('ISPY_clinical_clean.csv');
df.dropna(axis=0, inplace=True)

# func
def heading(str_):
    print("")
    print(10*"-" + str_ + 10*"-")
def footer():
    print(10*"==")
def con_table():
    print("Contingency Table")
def p_print(p_val, var_,out_, OR):
    print("The p-value for the effect of " + var_ + " on "+out_+" is:" + '{}'.format(p_val))
    print("The odds ratio for the effect of " + var_ + " on "+out_+" is:" + '{}'.format(OR))
    
#rename columns
df = df.rename(columns={'ERpos':'ER+', 
                        'PgRpos':'PR+',
                        'HR Pos':'HR+',
                        'BilateralCa':"Bilateral",
                        'sstat':'Survival',
                        'MRI LD Baseline':'MRI_LD_Baseline',
                        'MRI LD 1-3dAC':'MRI_LD_3days_after_chemo',
                        'MRI LD InterReg':'MRI_LD_Int_Reg'})

# Rename clinical outcomes and predictors 0/1 to make it easier to display
categorical_vars = ['ER+','PR+','HR+','Bilateral','PCR']
for str_ in categorical_vars:
    df[str_] = df[str_].replace([1,0],['Yes','No'])

# rename other predictors and outcomes
df.Survival = df.Survival.replace([7,8,9], ['Alive','Dead','Lost'])
df.Laterality = df.Laterality.replace([1,2],['Left','Right'])

# remove patients lost to follow up
df = df.loc[df.Survival != 'Lost',:]

# Function for Chi2
def my_chi2(outcome):
    categorical_predictors = df.columns[2:7]
    print("Reminder: The null hypothesis for chi-square tes is that NO association exists between the two variables.")
    print(outcome)
    for var in categorical_predictors:
        str_ = "effect of " + var + " on " + outcome
        heading(str_)
        T = pd.crosstab(df[var], 
                        df[outcome]).astype(float);
        chi2, p, dof, ex  = chi2_contingency( T.values )
        odds_ratio, _  = fisher_exact(T.values.T)
        print(T)
        p_print(p, var, outcome,odds_ratio)
        print()

# ANOVA age vs pCR
import seaborn as sns
from scipy import stats

def fig_01(ydata,xdata, hue_data):
    sns.boxplot(x= xdata, y=ydata, hue = hue_data, data=df, palette="Set3");

def anova_(predictor):
    print('ANOVA for ' + predictor + ' on PCR')
    f, p = stats.f_oneway(df.loc[df.PCR=='No',:][predictor], 
                                                    df.loc[df.PCR=='Yes',:][predictor])
                                                    
    print("The p-value for the effect of  " + predictor + " on PCR is:" + '{}'.format(p))
    print(10*'----')
    
    print('ANOVA for ' + predictor + ' on Survival')
    f, p = stats.f_oneway(df.loc[ df.Survival == 'Alive',:][predictor], df.loc[ df.Survival == 'Dead',:][predictor])
    
    print("The p-value for the effect of  " + predictor + " on Survival is:" + '{}'.format(p))
    print(10*'----')
