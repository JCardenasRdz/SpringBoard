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
  1. `Age` (Years)
  2. `Race`, encoded as:
  - 1 = Caucasian
  - 3 = African American
  - 4 = Asian
  - 5 = Native Hawaiian
  - 6 = American Indian
  - 50 = Multiple race
  3. Estrogen Receptor Status (`ER+`) encoded as:
  - 1 = Positive
  - 0 = Negative
  - Blank = Indeterminate
  4. Progesterone Receptor Status (`PR+`) encoded as:
  - 1 = Positive
  - 0 = Negative
  - Blank = Indeterminate
  5. Hormone Receptor Status (`ER+`)
  - 1 = Positive
  - 0 = Negative
  - Blank = Indeterminate
  6. Bilateral Breast Cancer (`Bilateral`):
  - 1 = Cancer Detected on both breasts
  - 0 = Cancer Detected in a single breast
  7. Breast with major or single Tumor (`Laterality`):
  - 1 = Left breast
  - 2 = Right breast
  8. Largest tumor dimension at Baseline estimated by MRI (`MRI_LD_Baseline`)
  - Continous variable
  9. Largest tumor dimension 1-3 days after NAC estimated by MRI (`MRI_LD_1_3dAC`)
  - Continous variable
  10. Largest tumor dimension between cycles of NAC estimated by MRI (`MRI_LD_Int_Reg`)
  - Continous variable
  11. Largest tumor dimension before surgery estimated by MRI (`MRI_LD_PreSurg`)
  - Continous variable

_*MRI with contrast at three time points:*_
- Modalities: 		MR, SEG
- Number of Patients: 222
- Number of Studies: 	847
- Number of Images: 	386,528
- Images Size (GB)	76.2 GigaByte

## Clinical Outcomes
Several clinical outcomes are described in this data set.
1. Survival Status (`Survival`):
- 7 = Alive
- 8 = Dead
- 9 = Lost to follow up
2. Length of Survival (`Survival_length`):
- Days from study entry to death or last follow-up
3. Recurrence-free survival (`RFS`):
- days from from NCAC start until progression or death
4. Recurrence-free survival indicator (`RFS_code`)
- progression or death (1),
- removed from survival curve (0)
5. Pathologic Complete Response (`PCR`) post-neoadjuvant ?:
- 1 = Yes
- 0 = No
- Lost (Blank)
6. Residual Cancer Burden class (`RCB`):
- 0 = RCB index (Class 0)
- 1 = RCB index less than or equal to 1.36 (Class I)
- 2 = RCB index greater than 1.36 or equal to 3.28  (Class II)
- 3 = III, RCB index greater than 3.28 (Class III)
- Blank = unavailable or no surgery
