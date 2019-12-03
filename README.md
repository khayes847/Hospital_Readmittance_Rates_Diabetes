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
  The target variable describes whether the patients are readmiited within 30 days, after more than 30 days, or not at all.
  The data contains 55 features. These include demographics, admission information (ex. patient id, discharge code, etc), and information regarding previous admissions. It also includes glucose serum and A1c test results (when the tests were performed), up to three diagnoses, and information on medications used. A full description of the features, including missing data, can be found at https://www.hindawi.com/journals/bmri/2014/781670/tab1.

- **Data Preparation**:
  We first separate the target variable, and change the target values to whether they were readmitted at all. We drop rows with null values when doing so won't lead to excessive data loss. In addition, we remove all instances where the same patient is hospitalized multiple times, or where the discharge code suggests that the patient would not be able to return due to either death or imprisonment. We also drop features that have excessive missing data or that provide no information.
  Next, we bin categorical data. For non-ordinal categorical data (such as race), we bin it with the purpose of creating bins as equal in size as possible, while ordinal data is binned to approximate a normal distribution. In addition, when data is presented using hospital code (diagonses, admissions, etc), we determine these assignations using https://www.findacode.com, and bin them accordingly. We create a feature summing each of the diagnoses by category, and features that sum the non-insulin medication data.
  After this, we create dummy variables for each of the categorical features. Finally, we bin some numeric data and create log distributions for others in order to remove outliers and to increase distribution normality.

- **Modeling**:
  To determine our baseline, I will vectorize the words in each comment, and perform a simple logistic regression using only the binary 'offensive' categorization as a target. For our final model, I will use a multinary target that reflects both offensiveness and the presence of 'identity group' content. In addition, this model will use a Keras deep learning algorithm.

- **Evaluation**:
  Since the inoffensive/offensive comment ratio is imbalanced, and since we have an interest in minimizing false 'offensive' assignations more than false 'inoffensive' assignations, I will use a custom cost function that penalizes/rewards each categorization appropriately. However, when searching for parameters, I will use the F1-score for the sake of simplicity to find the parameters that best balance our precision and recall. 

- **Deployment**:
  I will save and make available my final Keras model.

## Files in Repository:

* __README.md__ - A summary of the project.

* __technical_notebook.ipynb__ - Step-by-step walk through of the modeling/optimization process with rationale explaining decisions made. After cleaning and preparing the data, the notebook uses a gridsearch to create a logistic regression of the vectorized comments, and determines a baseline score using a custom cost function. For comparison, it also shows the score using only test data that contains 'identity' content. It then shows the process of creating a Keras deep-learning model, and compares the resulting scores (both for the entire test dataset and for only 'identity group' comments) to the prior model.

* __data_prep.py__ - Gathers and prepares data for analysis.

* __functions.py__ - Provides general functions used in the technical notebook, including Keras modeling and cost function calculation.

* __train_bias.csv__ - The raw dataset prior to data preparation.

* __target.csv__ - The cleaned target variables.

* __target.csv__ - The cleaned feature variables.

* __presentation.pptx__ - a presentation detailing the project and its results.

* __presentation.pdf__ - a presentation detailing the project and its results (in pdf form).
