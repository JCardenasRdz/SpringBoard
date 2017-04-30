# Captsone Project 01
## by Julio Cardenas-Rodriguez

## Description and Objectives
The goal of this project is to improve the prediction of clinical outcomes to neoadjuvant chemotherapy in patients with breast cancer.
Currently, most patients with breast cancer undergo neoadjuvant chemotherapy, which is aimed to reduce the tumor size (burden) before surgery to remove the tumor or the entire breast. Some of the patients response completely to the therapy and the patient does not present any residual tumor at the time of surgery. On the other hand, some patients have residual disease at the time of surgery and further treatment is required.  

## Data Source  
These data for **222 patients** treated for breast cancer was obtained from the cancer imaging archive and the Breast Imaging Research Program at UCSF, in collaboration with ACRIN, CALGB, the I-SPY TRIAL, and TCIA. More details [here](https://wiki.cancerimagingarchive.net/display/Public/ISPY1).  

## Predictors of clinical outcomes  
Two types of data are available for this project:   

_*Clinical Data as a XLS file with the following fields:*_
  1. Age (Years)
  2. Race, encoded as: 1=Caucasian, 3=African American, 4=Asian5=Native Hawaiian, 6=American Indian, 50=Multiple race
  3. Estrogen Receptor Status (ER) : 1 (Positive), 0 (Negative), Blank (Indeterminate)
  4. Progesterone Receptor Status (ER) : 1 (Positive), 0 (Negative), Blank (Indeterminate)
  5. Hormone Receptor Status (ER) : 1 (Positive), 0 (Negative), Blank (Indeterminate)
  6. Bilateral Breast Cancer (both breasts): 1 (Both), 0 (Single)
  7. Breast with major or single Tumor: 1 (Left), 2(Right)
  8. Tumor Volume at Baseline
  9. Tumor Volume Right after NAC
  10. Tumor Volume Between cycles:
  11. Tumor Volume before surgery   

_*MRI with contrast at three time points:*_
- Modalities: 		MR, SEG
- Number of Patients: 222
- Number of Studies: 	847
- Number of Images: 	386,528
- Images Size (GB)	76.2 GigaByte

## Clinical Outcomes
# Data Wrangling Captsone Project 01

## Description and Objectives
The goal of this project is to improve the prediction of clinical outcomes to neoadjuvant chemotherapy in patients with breast cancer.
Currently, most patients with breast cancer undergo neoadjuvant chemotherapy, which is aimed to reduce the tumor size (burden) before surgery to remove the tumor or the entire breast. Some of the patients response completely to the therapy and the patient does not present any residual tumor at the time of surgery. On the other hand, some patients have residual disease at the time of surgery and further treatment is required.  

## Data Source  
These data for **222 patients** treated for breast cancer was obtained from the cancer imaging archive and the Breast Imaging Research Program at UCSF, in collaboration with ACRIN, CALGB, the I-SPY TRIAL, and TCIA. More details [here](https://wiki.cancerimagingarchive.net/display/Public/ISPY1).  

## Predictors of clinical outcomes  
Two types of data are available for this project:   

_*Clinical Data as a XLS file with the following fields:*_
  1. Age (Years)
  2. Race, encoded as: 1=Caucasian, 3=African American, 4=Asian5=Native Hawaiian, 6=American Indian, 50=Multiple race
  3. Estrogen Receptor Status (ER) : 1 (Positive), 0 (Negative), Blank (Indeterminate)
  4. Progesterone Receptor Status (ER) : 1 (Positive), 0 (Negative), Blank (Indeterminate)
  5. Hormone Receptor Status (ER) : 1 (Positive), 0 (Negative), Blank (Indeterminate)
  6. Bilateral Breast Cancer (both breasts): 1 (Both), 0 (Single)
  7. Breast with major or single Tumor: 1 (Left), 2(Right)
  8. Tumor Volume at Baseline
  9. Tumor Volume Right after NAC
  10. Tumor Volume Between cycles:
  11. Tumor Volume before surgery   

_*MRI with contrast at four time points (same than the volumes):*_
- Modalities: 		MR, SEG
- Number of Patients: 222
- Number of Studies: 	847
- Number of Images: 	386,528
- Images Size (GB)	76.2 GigaByte

## Clinical Outcomes
Several clinical outcomes are described in this data set.
- Survival Status:
    - Alive (7)
    - Dead (8)
    - Lost to follow up (9)
- Length of Survival: days from study entry to death or last follow-up
- Recurrence-free survival (RFS): days from from NCAC start until progression or death
- RFS indicator: Recurrence-free survival indicator
    - progression or death (1),
    - removed from survival curve (0)
- Pathologic Complete Response (pCR) post-neoadjuvant:
    - Yes (1)
    - No (0)
    - Lost (Blank)
- Residual Cancer Burden class:
    - 0 = RCB index (Class 0)
    - 1 = RCB index less than or equal to 1.36 (Class I)
    - 2 = RCB index greater than 1.36 or equal to 3.28  (Class II)
    - 3 = III, RCB index greater than 3.28 (Class III)
    - Blank = unavailable or no surgery

# Data Wrangling: Clinical data
As mentioned above the clinical data (other than images) comes in an excel file with multiple tabs.
The objective for data cleaning are to: 1) remove columns we don't need, 2) Encode categorical and ordinal variables as needed, 3) merge all data into a single CSV for future analysis.
## Python Code
``` Python
import pandas as pd
file = './data/I-SPY_1_All_Patient_Clinical_and_Outcome_Data.xlsx'

# load and set index of predictors
predictors = pd.read_excel(file, sheetname='predictors')
predictors = predictors.set_index('SUBJECTID')

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
```
Please note that all blanks spaces in the excel file are turned into NaN entries in the final data set. I decided not to remove al NaN to preserved as much decide as possible given that each clinical outcome has a different number of NaN entries
