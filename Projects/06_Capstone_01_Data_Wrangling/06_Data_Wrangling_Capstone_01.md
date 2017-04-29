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

_*MRI with contrast at three time points:*_ 
- Modalities: 		MR, SEG
- Number of Patients: 222
- Number of Studies: 	847
- Number of Images: 	386,528
- Images Size (GB)	76.2 GigaByte

## Clinical Outcomes
