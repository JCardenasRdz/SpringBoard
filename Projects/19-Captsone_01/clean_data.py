import pandas as pd
file = './data/I-SPY_1_All_Patient_Clinical_and_Outcome_Data.xlsx'

# load and set index of predictors
predictors = pd.read_excel(file, sheetname='predictors')
predictors = predictors.set_index('SUBJECTID')

# drop NaN
#ISPY.dropna(inplace=True)

# drop Columns I don't need
predictors.drop(['DataExtractDt','Her2MostPos','HR_HER2_CATEGORY','HR_HER2_STATUS'],axis=1,inplace=True)
#encode race and drop initial variable
predictors = predictors.join(pd.get_dummies(predictors['race_id'], prefix=['Race']))
predictors.drop(['race_id'], axis=1,inplace=True)

# load predictors and drop columns I don't need
outcomes_df = pd.read_excel(file, sheetname='outcomes')
outcomes_df.drop(['DataExtractDt'],axis=1,inplace=True)
#
outcomes_df = outcomes_df.set_index('SUBJECTID')

#merge PCR and predictors using the Subject ID index
ISPY = predictors.join(outcomes_df)

# save clean data as CSV
ISPY.to_csv('./data/ISPY_clinical_clean.csv')
