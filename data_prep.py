"""
This file contains functions for use in data preparation.
"""

import pandas as pd
import numpy as np
import swifter
import operator


def upload():
    """Uploads the necessary data and returns the y
    and X databases."""
    print('Upload')
    data = pd.read_csv('data/diabetic_data.csv')
    data = data.set_index('encounter_id')
    X_val = data.drop(columns=['readmitted'])
    y_val = data.readmitted
    return X_val, y_val


def y_clean(data):
    """Replaces readmitted data with cleaned values"""
    print('y_clean')
    data = data.replace({'NO': 'no', '>30': 'over_30_days', '<30': 'under_30_days'})
    return data


def column_drop(data):
    """Drops columns that can't provide information due to either missing data or 
    lack of variance"""
    print('column_drop')
    data = data.drop(columns=['weight', 'payer_code', 'medical_specialty', 'examide',
                              'citoglipton', 'metformin-rosiglitazone', 'patient_nbr'])
    return data


def null_value_drop(data):
    """Drops rows will null data where it will not lead to excessive data loss"""
    print('null_value_drop')
    data = data.loc[data.race != '?']
    data = data.loc[data.gender != 'Unknown/Invalid']
    data = data.loc[~((data.diag_1 == '?')&(data.diag_2 == '?')&(data.diag_3 == '?'))]
    return data


def x_clean(data):
    """Cleans string values in X database"""
    print('x_clean')
    data['race'] = data.race.replace({'AfricanAmerican': 'african_american'})
    data['max_glu_serum'] = data.max_glu_serum.replace({'>200': '200_to_300',
                                                        '>300': 'more_than_300'})
    data['age'] = data.age.replace({'[0-10)': 1, '[10-20)': 2,
                                    '[20-30)': 3, '[30-40)': 4,
                                    '[40-50)': 5, '[50-60)': 6,
                                    '[60-70)': 7, '[70-80)': 8,
                                    '[80-90)': 9, '[90-100)': 10})
    data['A1Cresult'] = data.A1Cresult.replace({'>7': '7_to_8', ">8": "over_8"})
    return data


def reset_indices(X_val, y_val):
    """Ensures that dropped rows are reflected in y dataset.
    Resets indices for X and y datasets"""
    print('reset_indices')
    X_index = list(X_val.index)
    y_val = y_val.loc[y_val.index.isin(X_index)==True]
    return X_val, y_val


def values_lower(data):
    """Makes all string values lowercase"""
    print('values_lower')
    data_str_cols = list((data.select_dtypes(include=['object'])).columns)
    for col in data_str_cols:
        data[col] = data[col].apply(lambda xii: xii.lower())
    return data


def column_lowercase(data):
    """Makes all column names lowercase"""
    print('column_lowercase')
    data_cols = list(data.columns)
    for col in data_cols:
        data = data.rename(columns={str(col): col.lower()})
    return data


def categorize_all(data):
    """Transforms diagnosis data for all three rows"""
    data['diag_1'] = data.diag_1.swifter.apply(lambda x: float(x[1:])+2000 if x[:1] == 'e'
                                               else (float(x[1:])+1000 if x[:1] == 'v' else
                                                     (0 if x[:1] == "?" else
                                                      (float(x)))))
    data['diag_2'] = data.diag_2.swifter.apply(lambda x: float(x[1:])+2000 if x[:1] == 'e'
                                               else (float(x[1:])+1000 if x[:1] == 'v' else
                                                     (0 if x[:1] == "?" else
                                                      (float(x)))))
    data['diag_3'] = data.diag_3.swifter.apply(lambda x: float(x[1:])+2000 if x[:1] == 'e'
                                               else (float(x[1:])+1000 if x[:1] == 'v' else
                                                     (0 if x[:1] == "?" else
                                                      (float(x)))))
    return data


def diagnoses_1_3(data):
    """Creates a column summing the number of diagnoses from each category"""
    data['icd_1_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=1 and x<140 else 0)
    data['icd_1_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=1 and x<140 else 0)
    data['icd_1_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=1 and x<140 else 0)
    data['icd_1'] = data.swifter.apply(lambda row: row.icd_1_1 + row.icd_1_2 + row.icd_1_3, axis=1)
    data['icd_2_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=140 and x<240 else 0)
    data['icd_2_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=140 and x<240 else 0)
    data['icd_2_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=140 and x<240 else 0)
    data['icd_2'] = data.swifter.apply(lambda row: row.icd_2_1 + row.icd_2_2 + row.icd_2_3, axis=1)
    data['icd_3_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=240 and x<280 else 0)
    data['icd_3_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=240 and x<280 else 0)
    data['icd_3_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=240 and x<280 else 0)
    data['icd_3'] = data.swifter.apply(lambda row: row.icd_3_1 + row.icd_3_2 + row.icd_3_3, axis=1)
    data['icd_4_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=280 and x<290 else 0)
    data['icd_4_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=280 and x<290 else 0)
    data['icd_4_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=280 and x<290 else 0)
    data['icd_4'] = data.swifter.apply(lambda row: row.icd_4_1 + row.icd_4_2 + row.icd_4_3, axis=1)
    data['icd_5_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=290 and x<320 else 0)
    data['icd_5_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=290 and x<320 else 0)
    data['icd_5_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=290 and x<320 else 0)
    data['icd_5'] = data.swifter.apply(lambda row: row.icd_5_1 + row.icd_5_2 + row.icd_5_3, axis=1)
    data['icd_6_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=320 and x<390 else 0)
    data['icd_6_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=320 and x<390 else 0)
    data['icd_6_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=320 and x<390 else 0)
    data['icd_6'] = data.swifter.apply(lambda row: row.icd_6_1 + row.icd_6_2 + row.icd_6_3, axis=1)
    data['icd_7_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=390 and x<460 else 0)
    data['icd_7_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=390 and x<460 else 0)
    data['icd_7_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=390 and x<460 else 0)
    data['icd_7'] = data.swifter.apply(lambda row: row.icd_7_1 + row.icd_7_2 + row.icd_7_3, axis=1)
    data = data.drop(columns = ['icd_1_1', 'icd_1_2', 'icd_1_3', 'icd_2_1', 'icd_2_2',
                                'icd_2_3', 'icd_3_1', 'icd_3_2', 'icd_3_3', 'icd_4_1',
                                'icd_4_2', 'icd_4_3', 'icd_5_1', 'icd_5_2', 'icd_5_3',
                                'icd_6_1', 'icd_6_2', 'icd_6_3', 'icd_7_1', 'icd_7_2',
                                'icd_7_3'])
    return data


def diagnoses_2_3(data):
    """Creates a column summing the number of diagnoses from each category"""
    data['icd_8_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=460 and x<520 else 0)
    data['icd_8_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=460 and x<520 else 0)
    data['icd_8_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=460 and x<520 else 0)
    data['icd_8'] = data.swifter.apply(lambda row: row.icd_8_1 + row.icd_8_2 + row.icd_8_3, axis=1)
    data['icd_9_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=520 and x<580 else 0)
    data['icd_9_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=520 and x<580 else 0)
    data['icd_9_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=520 and x<580 else 0)
    data['icd_9'] = data.swifter.apply(lambda row: row.icd_9_1 + row.icd_9_2 + row.icd_9_3, axis=1)
    data['icd_10_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=580 and x<630 else 0)
    data['icd_10_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=580 and x<630 else 0)
    data['icd_10_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=580 and x<630 else 0)
    data['icd_10'] = data.swifter.apply(lambda row: row.icd_10_1 + row.icd_10_2 + row.icd_10_3, axis=1)
    data['icd_11_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=630 and x<680 else 0)
    data['icd_11_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=630 and x<680 else 0)
    data['icd_11_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=630 and x<680 else 0)
    data['icd_11'] = data.swifter.apply(lambda row: row.icd_11_1 + row.icd_11_2 + row.icd_11_3, axis=1)
    data['icd_12_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=680 and x<710 else 0)
    data['icd_12_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=680 and x<710 else 0)
    data['icd_12_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=680 and x<710 else 0)
    data['icd_12'] = data.swifter.apply(lambda row: row.icd_12_1 + row.icd_12_2 + row.icd_12_3, axis=1)
    data['icd_13_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=710 and x<740 else 0)
    data['icd_13_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=710 and x<740 else 0)
    data['icd_13_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=710 and x<740 else 0)
    data['icd_13'] = data.swifter.apply(lambda row: row.icd_13_1 + row.icd_13_2 + row.icd_13_3, axis=1)
    data['icd_14_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=740 and x<760 else 0)
    data['icd_14_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=740 and x<760 else 0)
    data['icd_14_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=740 and x<760 else 0)
    data['icd_14'] = data.swifter.apply(lambda row: row.icd_14_1 + row.icd_14_2 + row.icd_14_3, axis=1)
    data = data.drop(columns = ['icd_8_1', 'icd_8_2', 'icd_8_3', 'icd_9_1', 'icd_9_2',
                                'icd_9_3', 'icd_10_1', 'icd_10_2', 'icd_10_3', 'icd_11_1',
                                'icd_11_2', 'icd_11_3', 'icd_12_1', 'icd_12_2', 'icd_12_3',
                                'icd_13_1', 'icd_13_2', 'icd_13_3', 'icd_14_1', 'icd_14_2',
                                'icd_14_3'])
    return data


def diagnoses_3_3(data):
    """Creates a column summing the number of diagnoses from each category"""
    data['icd_15_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=760 and x<780 else 0)
    data['icd_15_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=760 and x<780 else 0)
    data['icd_15_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=760 and x<780 else 0)
    data['icd_15'] = data.swifter.apply(lambda row: row.icd_15_1 + row.icd_15_2 + row.icd_15_3, axis=1)
    data['icd_16_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=780 and x<800 else 0)
    data['icd_16_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=780 and x<800 else 0)
    data['icd_16_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=780 and x<800 else 0)
    data['icd_16'] = data.swifter.apply(lambda row: row.icd_16_1 + row.icd_16_2 + row.icd_16_3, axis=1)
    data['icd_17_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=800 and x<1000 else 0)
    data['icd_17_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=800 and x<1000 else 0)
    data['icd_17_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=800 and x<1000 else 0)
    data['icd_17'] = data.swifter.apply(lambda row: row.icd_17_1 + row.icd_17_2 + row.icd_17_3, axis=1)
    data['icd_18_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=1000 and x<1100 else 0)
    data['icd_18_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=1000 and x<1100 else 0)
    data['icd_18_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=1000 and x<1100 else 0)
    data['icd_18'] = data.swifter.apply(lambda row: row.icd_18_1 + row.icd_18_2 + row.icd_18_3, axis=1)
    data['icd_19_1'] = data['diag_1'].swifter.apply(lambda x: 1 if x>=2000 else 0)
    data['icd_19_2'] = data['diag_2'].swifter.apply(lambda x: 1 if x>=2000 else 0)
    data['icd_19_3'] = data['diag_3'].swifter.apply(lambda x: 1 if x>=2000 else 0)
    data['icd_19'] = data.swifter.apply(lambda row: row.icd_19_1 + row.icd_19_2 + row.icd_19_3, axis=1)
    data = data.drop(columns = ['icd_15_1', 'icd_15_2', 'icd_15_3', 'icd_16_1', 'icd_16_2',
                                'icd_16_3', 'icd_17_1', 'icd_17_2', 'icd_17_3', 'icd_18_1',
                                'icd_18_2', 'icd_18_3', 'icd_19_1', 'icd_19_2', 'icd_19_3'])
    return data


def number_meds(data, col_list):
    """Creates column with number of medications"""
    for col in col_list:
        data[f"{col}_use"] = data[col].swifter.apply(lambda x: 0 if x == 'no' else 1)
    data['num_meds'] = data.iloc[:, 63:83].sum(axis=1)
    data = data.drop(columns=['metformin_use', 'repaglinide_use', 'nateglinide_use',
                              'chlorpropamide_use', 'glimepiride_use', 'acetohexamide_use',
                              'glipizide_use', 'glyburide_use', 'tolbutamide_use',
                              'pioglitazone_use', 'rosiglitazone_use', 'acarbose_use',
                              'miglitol_use', 'troglitazone_use', 'tolazamide_use',
                              'insulin_use', 'glyburide-metformin_use',
                              'glipizide-metformin_use', 'glimepiride-pioglitazone_use',
                              'metformin-pioglitazone_use'])
    return data


def num_down(data, col_list):
    """Creates column with number of medications decreased"""
    for col in col_list:
        data[f"{col}_down"] = data[col].swifter.apply(lambda x: 1 if x == 'down' else 0)
    data['num_down'] = data.iloc[:, 64:].sum(axis=1)
    data = data.drop(columns=['metformin_down', 'repaglinide_down', 'nateglinide_down',
                          'chlorpropamide_down', 'glimepiride_down', 'acetohexamide_down',
                          'glipizide_down', 'glyburide_down', 'tolbutamide_down',
                          'pioglitazone_down', 'rosiglitazone_down', 'acarbose_down',
                          'miglitol_down', 'troglitazone_down', 'tolazamide_down', 'insulin_down',
                          'glyburide-metformin_down', 'glipizide-metformin_down',
                          'glimepiride-pioglitazone_down', 'metformin-pioglitazone_down'])
    return data


def num_up(data, col_list):
    """Creates column with number of medications increased"""
    for col in col_list:
        data[f"{col}_up"] = data[col].swifter.apply(lambda x: 1 if x == 'up' else 0)
    data['num_up'] = data.iloc[:, 65:].sum(axis=1)
    data = data.drop(columns=['metformin_up', 'repaglinide_up',
                          'nateglinide_up', 'chlorpropamide_up', 'glimepiride_up',
                          'acetohexamide_up', 'glipizide_up', 'glyburide_up', 'tolbutamide_up',
                          'pioglitazone_up', 'rosiglitazone_up', 'acarbose_up', 'miglitol_up',
                          'troglitazone_up', 'tolazamide_up', 'insulin_up',
                          'glyburide-metformin_up', 'glipizide-metformin_up',
                          'glimepiride-pioglitazone_up', 'metformin-pioglitazone_up'])
    return data
    

def med_columns(data):
    """Creates all new columns related to medication quantity and
    change"""
    col_list = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-pioglitazone']
    data = number_meds(data, col_list)
    data = num_down(data, col_list)
    data = num_up(data, col_list)
    return data



def clean_1_5():
    """Runs all cleaning functions"""
    X_1, y_1 = upload()
    y_2 = y_clean(y_1)
    X_2 = column_drop(X_1)
    X_3 = null_value_drop(X_2)
    X_4 = x_clean(X_3)
    X_5 = values_lower(X_4)
    X_6 = column_lowercase(X_5)
    X_7 = categorize_all(X_6)
    X_7.to_csv('./data/X_1_5.csv', index=None)
    y_2.to_csv('./data/y_cleaned.csv', index=None)
    return X_7, y_2


def clean_2_5(X_7):
    """Runs all cleaning functions"""
    X_8 = diagnoses_1_3(X_7)
    X_8.to_csv('./data/X_2_5.csv', index=None)
    return X_8


def clean_3_5(X_8):
    """Runs all cleaning functions"""
    X_9 = diagnoses_2_3(X_8)
    X_9.to_csv('./data/X_3_5.csv', index=None)
    return X_9

 
def clean_4_5(X_9):
    """Runs all cleaning functions"""
    X_10 = diagnoses_3_3(X_9)
    X_10.to_csv('./data/X_4_5.csv', index=None)
    return X_10


def clean_5_5(X_10, y_2):
    """Runs all cleaning functions"""
    X_11 = med_columns(X_10)
    X_12, y_3 = reset_indices(X_11, y_2)
    X_12.to_csv('./data/X_cleaned.csv', index=None)
    y_3.to_csv('./data/y_cleaned.csv', index=None)
    return X_12, y_3
     