# Hospital_Readmittance_Rates_In_Diabetic_Patients

## Created By:

* __Kyle Hayes__, __Kaleb Nyquist__

## Project Details:

30.3 million Americans, roughly 9.4% of the population, are diabetic. However, this population accounts for roughly 25% of all hospital admissions, including repeat admissions. Repeated hospital admissions are often a sign either of worsening complications due to diabetes, or of insufficient initial care. The purpose of this study is to create several models whose purpose is to predict whether patients will require a repeat hospital admission.

## Process Breakdown:

- **Business Understanding**:
  Hospital admission costs for persons with diabetes reached $124 billion in 2012 alone, placing a huge financial burden on patients. In addition, repeated hospital admissions also place a large burden on hospitals, both in quantity of patients, and in the severity of the operations that doctors must perform due to preventable illnesses. Both hospitals and patients have a large incentive to minimize the need for repeated hospitalizations.

- **Data Understanding**:
  We obtained the data from UCI's machine learning repository (https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008#). It consists of 100 thousand hospitalizations listing diabetes was listed as a diagnosis. These were collected from 1998-2008 from 130 hospitals.
  The target variable describes whether the patients are readmitted within 30 days, after more than 30 days, or not at all.
  The data contains 55 features. These include demographics, admission information (ex. patient id, discharge code, etc.), and information regarding previous admissions. It also includes glucose serum and A1c test results (when the tests were performed), up to three diagnoses, and information on medications used. A full description of the features, including missing data, can be found at https://www.hindawi.com/journals/bmri/2014/781670/tab1.

- **Data Preparation**:
  We first separate the target variable, and change the target values to whether they were readmitted at all. We drop rows with null values when doing so won't lead to excessive data loss. In addition, we remove all instances where the same patient is hospitalized multiple times, or where the discharge code suggests that the patient would not be able to return due to either death or imprisonment. We also drop features that have excessive missing data or that provide no information.
  Next, we bin categorical data. For non-ordinal categorical data (such as race), we bin it with the purpose of creating bins as equal in size as possible, while ordinal data is binned to approximate a normal distribution. In addition, when data is presented using hospital code (diagnoses, admissions, etc), we determine these assignations using https://www.findacode.com, and bin them accordingly. We create a feature summing each of the diagnoses by category, and features that sum the non-insulin medication data.
  After this, we create dummy variables for each of the categorical features. Finally, we bin some numeric data and create log distributions for others in order to remove outliers and to increase distribution normality.

- **Modeling**:
  To determine our baseline, we create a dummy classifier to serve as our baseline model. We then use the data to create a logistic regression, a random forests classifier, an AdaBoost classifier, and a gradient boosting classifier. For each of these, we split the data into a different stratified training (75%) and test (25%) dataset, and standardize the features to ensure that they are using the same scale. After this, we use a grid search to find the best parameters for each.

- **Evaluation**:
  Since we have a greater interest in creating a model that minimizes incorrect assignations of "no return admission" than we have in creating a model that minimizes incorrect assignations of "return admission", we will use the recall score of each model as our evaluation metric. For comparison's sake, we will also populate each model's accuracy and F1 score, and create a confusion matrix for each. For additional comparison, since the target values are imbalanced, we will create a precision-recall curve for each model instead of an ROC curve, and we will determine the AUC for each.

- **Deployment**:
  I will save and make available my final models.

## Files in Repository:

* __README.md__ - A summary of the project.

* __technical_notebook.ipynb__ - Step-by-step walk through of the modeling/optimization process with rationale explaining decisions made. After cleaning and preparing the data, the notebook creates and compares a dummy categorizer, a logistic regression, a random forests classifier, an AdaBoost classifier, and a gradient boosting classifier, using the description listed in 'Modeling'. It then graphs the recall score for each model.

* __data_prep.py__ - Gathers and prepares data for analysis.

* __functions.py__ - Provides functions used in the technical notebook.

* __presentation.pptx__ - A presentation detailing the project and its results.

* __presentation.pdf__ - A presentation detailing the project and its results (in pdf form).

* __images__ - Folder of images used in presentation.

* __models__ - Folder of pickled models used.

* __git_ignore__ - Prevents Git from uploading data folder.
