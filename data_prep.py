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
    X = data.drop(columns=['readmitted'])
    y = data.readmitted
    return X, y


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
                              'citoglipton', 'metformin-rosiglitazone'])
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
    data['age'] = data.age.replace({'[0-10)': '0_10', '[10-20)': '10_20',
                                    '[20-30)': '20_30', '[30-40)': '30_40',
                                    '[40-50)': '40_50', '[50-60)': '50_60',
                                    '[60-70)': '60_70', '[70-80)': '70_80',
                                    '[80-90)': '80_90', '[90-100)': '90_100'})
    data['A1Cresult'] = data.A1Cresult.replace({'>7': '7_to_8', ">8": "over_8"})
    return data


def reset_indices(X_val, y_val):
    """Ensures that dropped rows are reflected in y dataset.
    Resets indices for X and y datasets"""
    print('reset_indices')
    X_index = list(X_val.index)
    y = y_val.loc[y_val.index.isin(X_index)==True]
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
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


def categorize_float(data, column):
    """Transforms diagnosis data to integer"""
    diag_type = pd.Series([])
    for i in range(len(data)):
        print(round(float(i/len(data))*100, 2))
        if data[column][i][:1] == 'e':
            data_num = float(data[column][i][1:])
            diag_type[i] = data_num+2000
        elif data[column][i][:1] == 'v':
            data_num = float(data[column][i][1:])
            diag_type[i] = data_num+1000
        elif data[column][i][:1] == '?':
            diag_type[i] = 0
        else:
            diag_type[i] = float(data[column][i])
    data[column] = diag_type
    return data


def categorize_all(data):
    """Transforms diagnosis data for all three rows"""
    print('Column diag_1')
    data = categorize_float(data, 'diag_1')
    print('Column diag_2')
    data = categorize_float(data, 'diag_2')
    print('Column diag_3')
    data = categorize_float(data, 'diag_3')
    return data


def age_upper(data):
    """Creates column with integer for upper age range limit"""
    print('age_upper')
    age_upper = pd.Series([])
    for i in range(len(data)):
        print(round(float(i/len(data))*100, 2))
        if data['age'][i] == '0_10':
            age_upper[i] = 10
        elif data['age'][i] == '10_20':
            age_upper[i] = 20
        elif data['age'][i] == '20_30':
            age_upper[i] = 30
        elif data['age'][i] == '30_40':
            age_upper[i] = 40
        elif data['age'][i] == '40_50':
            age_upper[i] = 50
        elif data['age'][i] == '50_60':
            age_upper[i] = 60
        elif data['age'][i] == '60_70':
            age_upper[i] = 70
        elif data['age'][i] == '70_80':
            age_upper[i] = 80
        elif data['age'][i] == '80_90':
            age_upper[i] = 90
        else:
            age_upper[i] = 100
    data['age_upper'] = age_upper
    return data


def diag_range(data, lower, upper):
    """Creates series that adds up the number of diagnoses in a range"""
    diag_range_count = pd.Series([])
    for i in range(len(data)):
        print(round(float(i/len(data))*100, 2))
        total = 0
        if data['diag_1'][i] >= lower and data['diag_1'][i] < upper:
            total += 1
        if data['diag_2'][i] >= lower and data['diag_2'][i] < upper:
            total += 1
        if data['diag_3'][i] >= lower and data['diag_3'][i] < upper:
            total += 1
        diag_range_count[i] = total
    return diag_range_count


def diagnoses_1_3(data):
    """Creates a column summing the number of diagnoses from each category"""
    print('icd_1')
    data['icd_1'] = diag_range(data, lower=1, upper=140)
    print('icd_2')
    data['icd_2'] = diag_range(data, lower=140, upper=240)
    print('icd_3')
    data['icd_3'] = diag_range(data, lower=240, upper=280)
    print('icd_4')
    data['icd_4'] = diag_range(data, lower=280, upper=290)
    print('icd_5')
    data['icd_5'] = diag_range(data, lower=290, upper=320)
    print('icd_6')
    data['icd_6'] = diag_range(data, lower=320, upper=390)
    print('icd_7')
    data['icd_7'] = diag_range(data, lower=390, upper=460)
    return data


# def diag_range2(row, lower, upper):
#     """Creates a column summing the number of diagnoses in a range"""
#     diag_num = 0
#     diag_num += 1 if row['diag_1']>=lower and row['diag_1']<upper
#     diag_num += 1 if row['diag_2']>=lower and row['diag_2']<upper
#     diag_num += 1 if row['diag_3']>=lower and row['diag_3']<upper
#     return int(diag_num)


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



# def diagnoses_3_3(data):
#     """Creates a column summing the number of diagnoses from each category"""
#     print('icd_15')
#     data['icd_15'] = diag_range(data, lower=760, upper=780)
#     print('icd_16')
#     data['icd_16'] = diag_range(data, lower=780, upper=800)
#     print('icd_17')
#     data['icd_17'] = diag_range(data, lower=800, upper=1000)
#     print('icd_18')
#     data['icd_18'] = diag_range(data, lower=1000, upper=1100)
#     print('icd_19')
#     data['icd_19'] = diag_range(data, lower=2000, upper=2100)
#     return data


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
    data = num_steady(data, col_list)
    data = num_up(data, col_list)
    return data



def clean_1_5():
    """Runs all cleaning functions"""
    X_1, y_1 = upload()
    y_2 = y_clean(y_1)
    X_2 = column_drop(X_1)
    X_3 = null_value_drop(X_2)
    X_4 = x_clean(X_3)
    X_5, y_3 = reset_indices(X_4, y_2)
    X_6 = values_lower(X_5)
    X_7 = column_lowercase(X_6)
    X_8 = categorize_all(X_7)
    X_9 = age_upper(X_8)
    X_9.to_csv('./data/X_1_5.csv', index=None)
    y_3.to_csv('./data/y_cleaned.csv', index=None)
    return X_9, y_3


def clean_2_5(X_9):
    """Runs all cleaning functions"""
    X_10 = diagnoses_1_3(X_9)
    X_10.to_csv('./data/X_2_5.csv', index=None)
    return X_10


def clean_3_5(X_10):
    """Runs all cleaning functions"""
    X_11 = diagnoses_2_3(X_10)
    X_11.to_csv('./data/X_3_5.csv', index=None)
    return X_11

 
def clean_4_5(X_11):
    """Runs all cleaning functions"""
    X_12 = diagnoses_3_3(X_11)
    X_12.to_csv('./data/X_4_5.csv', index=None)
    return X_12


def clean_5_5(X_12):
    """Runs all cleaning functions"""
    X_13 = med_columns(X_12)
    X_13.to_csv('./data/X_cleaned.csv', index=None)
    return X_13
     