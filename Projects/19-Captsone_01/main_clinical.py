# import custom modules wrote by julio
import importlib
from capstone_01 import clean_data
from capstone_01 import inferential_statistics

# reload modules without restartign the kernall (it drives me crazy)
importlib.reload(clean_data)
importlib.reload(inferential_statistics)

# load data and cleant it
file = './raw_data/clinical/I-SPY_1_All_Patient_Clinical_and_Outcome_Data.xlsx'
df = clean_data.clean_my_data(file)

# prepare data for inferential_statistics
df = inferential_statistics.prepare_data(df)

# assign all categorical predictors
predictors = df.columns[1:7]

# run chi-squared test on PCR as outcome
inferential_statistics.my_chi2('PCR', predictors, df)

# run chi-squared test on Survival as outcome
inferential_statistics.my_chi2('Survival', predictors, df)

# ANOVA PCR
inferential_statistics.anova_pcr('age', df)
inferential_statistics.anova_pcr('MRI_LD_Baseline', df)

# ANOVA Survival vs Age
inferential_statistics.anova_survival('age', df)

# Tumor volumes
inferential_statistics.anova_survival('MRI_LD_Baseline', df)
inferential_statistics.anova_survival('MRI LD PreSurg', df)
