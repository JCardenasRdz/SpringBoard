# **Capstone Project Proposal 01**
# _Insights into hospital readmission using medicare's public data bases_
####  by: Julio Cárdenas-Rodríguez
![medicare](https://govbooktalk.files.wordpress.com/2014/07/medicare-website-image.jpg)

## Background
In the United States, all people who are 65 years old or older receive healthcare insurance that is paid the beneficiary's contribution and taxes, this program is known as medicare. Medicare also covers certain younger people with disabilities. The program is structured in two parts:
- Part A to cover health care at medical facilities (mostly).
- Part B which covers the costs of health care outside medical facilities

The federal government maintains a data base of several quality metrics to compare hospitals, out-patient services, physicians, nursing homes, etc. In this project I will concentrate only on the hospitals. Medicare uses the following metrics/data bases to make funding/reimbursement decisions:
1. Survey of patients experiences
2. Timely and effective care
3. Complications
4. Readmissions and deaths
5. Use of medical imaging
6. Payment and value of care
7. Hospital Rating

## Goals
The department of health and human services makes the data about hospital quality publicly available. However, these data is presented in a website that is not user friendly and makes it impossible to support policy decisions based on these public data.
With this in mind, I want to address the following questions:

1. Is the readmission rate to the hospital affected by location and disease?
2. Is it possible to predict readmission rate ?
3. Is cost related to a lower readmission rate? 


## Data Source
I will use the entire hospital compare data base from 2012 to 2017, and have posted all raw csv files [@Data.World](https://data.world/julio). There are thirty one tables in the Hospital Compare database.
1. Agency for Healthcare and Research Quality – National.csv
2. Agency for Healthcare Research  and Quality – State.csv
3. Agency for Healthcare Research and Quality.csv
4. HCAHPS Measures – National.csv
5. HCAHPS Measures  – State.csv
6. HCAHPS Measures.csv
7. Healthcare_Associated_Infections.csv
8. Healthcare_Associated_Infections_State.csv  
9. Hospital Acquired Condition – National.csv
10. Hospital Acquired Condition.csv
11. Hospital_Data.csv
12. Measure Dates.csv
13. Medicare Payment and Volume Measures – National.csv
14. Medicare Payment and Volume Measures – State.csv
15. Medicare Payment and Volume Measures.csv
16. Medicare Spending per  Patient.csv
17. Outcome of Care Measures –National.csv
18. Outcome of Care Measures –State.csv
19. Outcome of Care Measures.csv
20. Outpatient Imaging Efficiency Measures – National.csv
21. Outpatient Imaging Efficiency Measures – State.csv
22. Outpatient Imaging Efficiency Measures.csv
23. Process of Care Measures –Children.csv
24. Process of Care Measures –Heart Attack.csv
25. Process of Care Measures – Heart Failure.csv
26. Process of Care Measures – National.csv
27. Process of Care Measures –Pneumonia.csv
28. Process of Care Measures – SCIP.csv
29. Process of Care Measures –State.csv
30. Structural Measures.csv
31. Readmission Reduction.csv

## Approach
1. Understand the variables in the data 
2. Clean data to exclude redundant entries
3. Explore and construct models
4. build API
5. Test API

## Deliverables
- Report/Paper
- Jupyter Notebooks with intermediate data analysis
- REST API to predict readmission rate.