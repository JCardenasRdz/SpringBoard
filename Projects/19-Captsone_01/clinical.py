
# import custom modules wrote by julio
import importlib
from capstone_01 import clean_data
from capstone_01 import inferential_statistics

# reload modules without restartign the kernall (it drives me crazy)
importlib.reload(clean_data);
importlib.reload(inferential_statistics);

# load data and cleant it
file = './raw_data/clinical/I-SPY_1_All_Patient_Clinical_and_Outcome_Data.xlsx'
df = clean_data.clean_my_data(file)


# ## Inferential_statistics
# ### 1. Categorical vs Categorical (Chi-2)

# assign all categorical predictors
predictors = df.columns[1:7]

# run chi-squared test on PCR as outcome
inferential_statistics.my_chi2('PCR', ['race'], df.loc[df.race != 0,:])

# run chi-squared test on Survival as outcome
inferential_statistics.my_chi2('Survival', predictors, df.loc[df.race != 0,:])


# ### 2. Continous vs Categorical (ANOVA)

# ANOVA PCR
#inferential_statistics.anova_pcr('age', df)
#inferential_statistics.anova_pcr('MRI_LD_Baseline', df)

# ANOVA Survival vs age
#inferential_statistics.anova_survival('age', df)

# Tumor volumes
mri_list = ['MRI_LD_Baseline', 'MRI_LD_1_3dAC', 'MRI_LD_Int_Reg', 'MRI_LD_PreSurg']
for time_point in mri_list:
    print( inferential_statistics.anova_survival(time_point, df) )
