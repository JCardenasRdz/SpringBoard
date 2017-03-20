#### A. Initial observations based on the plot above  
- Overall, rate of readmissions is trending down with increasing number of discharges.
- With lower number of discharges, there is a greater incidence of excess rate of readmissions (area shaded red)
- With higher number of discharges, there is a greater incidence of lower rates of readmissions (area shaded green)
> JCR:
a) This is an erroneous interpretation. A horizontal line does not indicate any trend. It means that there is no correlation between the two variables. In other words, one of them remains constant. While the other increases or decreases

#### B. Statistics  
- In hospitals/facilities with number of discharges < 100, mean excess readmission rate is 1.023 and 63% have excess readmission rate greater than 1
- In hospitals/facilities with number of discharges > 1000, mean excess readmission rate is 0.978 and 44% have excess readmission rate greater than 1
> b) These are totally arbitrary cutoffs.
  c) No statistical test is pretested to determine the significance of this "correlation".
  d) The data is not cross validated in any way.


#### C. Conclusions  
- There is a significant correlation between hospital capacity (number of discharges) and readmission rates.
- Smaller hospitals/facilities may be lacking necessary resources to ensure quality care and prevent complications that lead to readmissions.
> e) These conclusions are not supported by the analysis.

#### D. Regulatory policy recommendations  
- Hospitals/facilities with small capacity (< 300) should be required to demonstrate upgraded resource allocation for quality care to continue operation.
- Directives and incentives should be provided for consolidation of hospitals and facilities to have a smaller number of them with higher capacity and number of discharges.
> f) These recommendations are not supported by the analysis nor will help in any way.

### Recommendations
## Statistical
1. Use an statistical test to determine the degree of correlation between two given variables. Initially, a simple linear regression should be enough.
2. Answer the following questions:
    - Is ``Excess Readmission Ratio`` correlated with other continous variables?
    - Does state have an effect on the ``Excess Readmission Ratio`` ?

## Coding
3. Create a separate ``.py`` file with the code to make it easier to follow.
4. Annotate code to justify non obvious choices. For example _Why did you use [81:-3] only?_
5. Avoid redundant code such as :
~~~Python
clean_hospital_read_df.loc[:, 'Number of Discharges'] = clean_hospital_read_df['Number of Discharges'].astype(int)
~~~
6. Remove entries that are not real numbers (strings, etc.) for the continous data.
7. Check that all US states are present
