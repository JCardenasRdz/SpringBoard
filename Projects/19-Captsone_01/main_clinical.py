# import custom modules wrote by julio
import importlib
from capstone_01 import clean_data; importlib.reload(clean_data)
from capstone_01 import inferential_statistics;  importlib.reload(inferential_statistics) 

# load data and cleant it
file = './data/I-SPY_1_All_Patient_Clinical_and_Outcome_Data.xlsx'
df = clean_data.clean_my_data(file)

# prepare data for inferential_statistics
df = inferential_statistics.process_data(df)

# assign all categorical predictors
predictors = df.columns[0:7]

inferential_statistics.my_chi2('PCR',df)