import pandas as pd
pd.options.mode.chained_assignment = None

def clean_my_data(file):
    # load and set index of predictors
    predictors = pd.read_excel(file, sheetname='predictors')
    predictors = predictors.set_index('SUBJECTID')

    # drop Columns I don't need
    predictors.drop(['DataExtractDt','Her2MostPos','HR_HER2_CATEGORY','HR_HER2_STATUS'],axis=1,inplace=True)

#encode race and drop initial variable
#predictors = predictors.join(pd.get_dummies(predictors['race_id'], prefix=['Race']))
#predictors.drop(['race_id'], axis=1,inplace=True)

    # load predictors and drop columns I don't need
    outcomes_df = pd.read_excel(file, sheetname='outcomes')
    outcomes_df.drop(['DataExtractDt'],axis=1,inplace=True)
    outcomes_df = outcomes_df.set_index('SUBJECTID')

    #merge PCR and predictors using the Subject ID index
    ISPY = predictors.join(outcomes_df)

    ISPY = _organize_data(ISPY)
    return ISPY



def _organize_data(dataframe):
    # make a copy
    df = dataframe.copy()

    df.dropna(axis=0, inplace=True)
    #rename columns
    df = df.rename(columns={'ERpos':'ER+',
                        'PgRpos':'PR+',
                        'HR Pos':'HR+',
                        'BilateralCa':"Bilateral",
                        'sstat':'Alive',
                        'MRI LD Baseline':'MRI_LD_Baseline',
                        'MRI LD 1-3dAC':'MRI_LD_1_3dAC',
                        'MRI LD InterReg':'MRI_LD_Int_Reg',
                        'MRI LD PreSurg': 'MRI_LD_PreSurg',
                        'survDtD2 (tx)':'Survival_length',
                        'rfs_ind':'RFS_code',
                        'RCBClass':'RCB',
                        'Laterality':'Right_Breast',
                        'race_id':'White'})
    # Rename clinical outcomes and predictors 0/1 to make it easier to display
    categorical_vars = ['ER+','PR+','HR+','Bilateral','PCR']
    for str_ in categorical_vars:
        df[str_] = df[str_].replace([1,0],['Yes','No'])

    # rename other predictors and outcomes
    df.Alive = df.Alive.replace([7,8,9], ['Yes','No','Lost'])
    df.Right_Breast = df.Right_Breast.replace([1,2],['No','Yes'])

    df.White[df.White != 1] = 0
    df.White = df.White.replace([1,0],['Yes','No'])

    # remove patients lost to follow up
    df = df.loc[df.Alive != 'Lost',:]

    # output
    return df
