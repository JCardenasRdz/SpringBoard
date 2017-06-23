# Capstone 02 Ideas  

Magnetic Resonance Imaging (MRI) and computed tomography of X-rays (CT) have shown great success in providing anatomical insights into multiple diseases, but the vast majority of clinical studies and standards for clinical practice do not include quantitative metrics of MRI or CT. This makes the implementation of multi-center clinical trials particularly challenging, furthermore, even when research-level quantitative metrics are included, their variability makes them poor biomarkers for clinical trials and clinical practice.

This proposal seeks to address the limitations described above by pursuing a data-driven approach to improve medical imaging in three different areas.

#### 1. Machine-learning-based 3D segmentation of tumors
`Need`: the current anatomic unidimensional assessments of tumor burden such as largest tumor dimension(LTD) are insufficient to detect or predict response to therapeutical interventions.   
`Solution`: to compare the capacity of a) LTD and b) tumor _volume_ derived from machine-learning-based segmentation_ to detect or predict response to therapeutical interventions.   
`Data Source`: [The cancer imaging archive](http://www.cancerimagingarchive.net/)   


#### 2. Improved metrics for dynamic using new ma
`Need`: The current metrics of tumor permeability MRI have shown promise in single-site clinical trials but their implementation on multi-center clinical trials has failed because of the low repeatability and reproducibility  of these metrics.   
`Solution`: I have developed new algorithms for the analysis of dynamic data. I propose to use these new methods to analyze multi-center DSC and DCE-MRI data with the goal of identifying responders to therapy and/or improve patient segmentation.   
`Data Source`: [The cancer imaging archive](http://www.cancerimagingarchive.net/)  


#### 3. Comparison of medical images using perceptual hashing
`Need`: The analysis of medical images is mostly based on reports written by radiologists and not on quantitative metrics, furthermore, the detail and quality of radiological reports is highly dependent on the expertise and training of each radiologists. Thus a metric to automatically compare medical images is needed. This would provide radiologists with the capacity of mining medical imaging datasets not based on their size or report, but on how the images actually look.   
`Solution`: Adapt perceptual hashing for the analysis of 4D imaging datasets and determine its capacity to differentiate normal and abnormal CT and MRI scans.   
`Data Source`: [The cancer imaging archive](http://www.cancerimagingarchive.net/)  
