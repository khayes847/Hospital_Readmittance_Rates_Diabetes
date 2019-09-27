"""
This file contains functions for use in data preparation.
"""


import pandas as pd


def upload():
    """Uploads the necessary data and returns the y
    and X databases."""
    print('Upload')
    data = pd.read_csv('data/diabetic_data.csv')
    data = data.set_index('encounter_id')
    x_val = data.drop(columns=['readmitted'])
    y_val = data.readmitted
    return x_val, y_val


def y_clean(data):
    """Replaces readmitted data with cleaned values"""
    print('y_clean')
    data = data.replace({'NO': 0, '>30': 1, '<30': 1})
    return data


def column_drop(data):
    """Drops columns that can't provide information due to
    either missing data or lack of variance"""
    print('column_drop')
    data = data.drop(columns=['weight', 'payer_code', 'medical_specialty',
                              'examide', 'citoglipton',
                              'metformin-rosiglitazone'])
    return data


def null_value_drop(data):
    """Drops rows will null data where it will not lead
    to excessive data loss"""
    print('null_value_drop')
    data = data.loc[data.race != '?']
    data = data.loc[data.gender != 'Unknown/Invalid']
    data = data.loc[~((data.diag_1 == '?') &
                      (data.diag_2 == '?') & (data.diag_3 == '?'))]
    return data


def x_clean(data):
    """Cleans string values in X database"""
    print('x_clean')
    data['race'] = data.race.replace({'AfricanAmerican': 'african_american'})
    data['max_glu_serum'] = data.max_glu_serum.replace({'>200': '200_to_300',
                                                        '>300':
                                                        'more_than_300'})
    data['age'] = data.age.replace({'[0-10)': 1, '[10-20)': 2,
                                    '[20-30)': 3, '[30-40)': 4,
                                    '[40-50)': 5, '[50-60)': 6,
                                    '[60-70)': 7, '[70-80)': 8,
                                    '[80-90)': 9, '[90-100)': 10})
    data['A1Cresult'] = data.A1Cresult.replace({'>7': '7_to_8',
                                                ">8": "over_8"})
    return data


def reset_indices(x_val, y_val):
    """Ensures that dropped rows are reflected in y dataset.
    Resets indices for X and y datasets"""
    print('reset_indices')
    x_index = list(x_val.index)
    y_val = y_val.loc[y_val.index.isin(x_index) == True]
    return x_val, y_val


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
    print('categorize_all')
    data['diag_1'] = data.diag_1.swifter.apply(lambda x: float(x[1:])+2000
                                               if x[:1] == 'e'
                                               else (float(x[1:])+1000
                                                     if x[:1] == 'v'
                                                     else (0 if x[:1] == "?"
                                                           else (float(x)))))
    data['diag_2'] = data.diag_2.swifter.apply(lambda x: float(x[1:])+2000
                                               if x[:1] == 'e'
                                               else (float(x[1:])+1000
                                                     if x[:1] == 'v'
                                                     else (0 if x[:1] == "?"
                                                           else (float(x)))))
    data['diag_3'] = data.diag_3.swifter.apply(lambda x: float(x[1:])+2000
                                               if x[:1] == 'e'
                                               else (float(x[1:])+1000
                                                     if x[:1] == 'v'
                                                     else (0 if x[:1] == "?"
                                                           else (float(x)))))
    return data


def diagnoses_1_3(data):
    """Creates a column summing the number of diagnoses from each category"""
    print('diagnoses_1_3')
    data['icd_1_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 140 > x >= 1 else 0)
    data['icd_1_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 140 > x >= 1 else 0)
    data['icd_1_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 140 > x >= 1 else 0)
    data['icd_1'] = data.swifter.apply(lambda row: row.icd_1_1
                                       + row.icd_1_2 + row.icd_1_3, axis=1)
    data['icd_2_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 240 > x >= 140 else 0)
    data['icd_2_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 240 > x >= 140 else 0)
    data['icd_2_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 240 > x >= 140 else 0)
    data['icd_2'] = data.swifter.apply(lambda row: row.icd_2_1 + row.icd_2_2
                                       + row.icd_2_3, axis=1)
    data['icd_3_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 280 > x >= 240 else 0)
    data['icd_3_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 280 > x >= 240 else 0)
    data['icd_3_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 280 > x >= 240 else 0)
    data['icd_3'] = data.swifter.apply(lambda row: row.icd_3_1 + row.icd_3_2
                                       + row.icd_3_3, axis=1)
    data['icd_4_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 290 > x >= 280 else 0)
    data['icd_4_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 290 > x >= 280 else 0)
    data['icd_4_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 290 > x >= 280 else 0)
    data['icd_4'] = data.swifter.apply(lambda row: row.icd_4_1 + row.icd_4_2
                                       + row.icd_4_3, axis=1)
    data['icd_5_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 320 > x >= 290 else 0)
    data['icd_5_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 320 > x >= 290 else 0)
    data['icd_5_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 320 > x >= 290 else 0)
    data['icd_5'] = data.swifter.apply(lambda row: row.icd_5_1 + row.icd_5_2
                                       + row.icd_5_3, axis=1)
    data['icd_6_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 390 > x >= 320 else 0)
    data['icd_6_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 390 > x >= 320 else 0)
    data['icd_6_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 390 > x >= 320 else 0)
    data['icd_6'] = data.swifter.apply(lambda row: row.icd_6_1 + row.icd_6_2
                                       + row.icd_6_3, axis=1)
    data['icd_7_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 460 > x >= 390 else 0)
    data['icd_7_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 460 > x >= 390 else 0)
    data['icd_7_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 460 > x >= 390 else 0)
    data['icd_7'] = data.swifter.apply(lambda row: row.icd_7_1
                                       + row.icd_7_2 + row.icd_7_3, axis=1)
    data = data.drop(columns=['icd_1_1', 'icd_1_2', 'icd_1_3',
                              'icd_2_1', 'icd_2_2', 'icd_2_3',
                              'icd_3_1', 'icd_3_2', 'icd_3_3',
                              'icd_4_1', 'icd_4_2', 'icd_4_3',
                              'icd_5_1', 'icd_5_2', 'icd_5_3',
                              'icd_6_1', 'icd_6_2', 'icd_6_3',
                              'icd_7_1', 'icd_7_2', 'icd_7_3'])
    return data


def diagnoses_2_3(data):
    """Creates a column summing the number of diagnoses from each category"""
    print('diagnoses_2_3')
    data['icd_8_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 520 > x >= 460 else 0)
    data['icd_8_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 520 > x >= 460 else 0)
    data['icd_8_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 520 > x >= 460 else 0)
    data['icd_8'] = data.swifter.apply(lambda row: row.icd_8_1
                                       + row.icd_8_2 + row.icd_8_3, axis=1)
    data['icd_9_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                   if 580 > x >= 520 else 0)
    data['icd_9_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                   if 580 > x >= 520 else 0)
    data['icd_9_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                   if 580 > x >= 520 else 0)
    data['icd_9'] = data.swifter.apply(lambda row: row.icd_9_1
                                       + row.icd_9_2 + row.icd_9_3, axis=1)
    data['icd_10_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 630 > x >= 580 else 0)
    data['icd_10_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 630 > x >= 580 else 0)
    data['icd_10_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 630 > x >= 580 else 0)
    data['icd_10'] = data.swifter.apply(lambda row: row.icd_10_1
                                        + row.icd_10_2 + row.icd_10_3, axis=1)
    data['icd_11_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 680 > x >= 630 else 0)
    data['icd_11_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 680 > x >= 630 else 0)
    data['icd_11_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 680 > x >= 630 else 0)
    data['icd_11'] = data.swifter.apply(lambda row: row.icd_11_1
                                        + row.icd_11_2 + row.icd_11_3, axis=1)
    data['icd_12_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 710 > x >= 680 else 0)
    data['icd_12_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 710 > x >= 680 else 0)
    data['icd_12_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 710 > x >= 680 else 0)
    data['icd_12'] = data.swifter.apply(lambda row: row.icd_12_1
                                        + row.icd_12_2 + row.icd_12_3, axis=1)
    data['icd_13_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 740 > x >= 710 else 0)
    data['icd_13_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 740 > x >= 710 else 0)
    data['icd_13_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 740 > x >= 710 else 0)
    data['icd_13'] = data.swifter.apply(lambda row: row.icd_13_1
                                        + row.icd_13_2 + row.icd_13_3, axis=1)
    data['icd_14_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 760 > x >= 740 else 0)
    data['icd_14_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 760 > x >= 740 else 0)
    data['icd_14_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 760 > x >= 740 else 0)
    data['icd_14'] = data.swifter.apply(lambda row: row.icd_14_1
                                        + row.icd_14_2 + row.icd_14_3, axis=1)
    data = data.drop(columns=['icd_8_1', 'icd_8_2', 'icd_8_3', 'icd_9_1',
                              'icd_9_2', 'icd_9_3', 'icd_10_1', 'icd_10_2',
                              'icd_10_3', 'icd_11_1', 'icd_11_2', 'icd_11_3',
                              'icd_12_1', 'icd_12_2', 'icd_12_3', 'icd_13_1',
                              'icd_13_2', 'icd_13_3', 'icd_14_1', 'icd_14_2',
                              'icd_14_3'])
    return data


def diagnoses_3_3(data):
    """Creates a column summing the number of diagnoses from each category"""
    print('diagnoses_3_3')
    data['icd_15_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 780 > x >= 760 else 0)
    data['icd_15_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 780 > x >= 760 else 0)
    data['icd_15_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 780 > x >= 760 else 0)
    data['icd_15'] = data.swifter.apply(lambda row: row.icd_15_1
                                        + row.icd_15_2 + row.icd_15_3, axis=1)
    data['icd_16_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 800 > x >= 780 else 0)
    data['icd_16_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 800 > x >= 780 else 0)
    data['icd_16_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 800 > x >= 780 else 0)
    data['icd_16'] = data.swifter.apply(lambda row: row.icd_16_1
                                        + row.icd_16_2 + row.icd_16_3, axis=1)
    data['icd_17_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 1000 > x >= 800 else 0)
    data['icd_17_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 1000 > x >= 800 else 0)
    data['icd_17_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 1000 > x >= 800 else 0)
    data['icd_17'] = data.swifter.apply(lambda row: row.icd_17_1 +
                                        row.icd_17_2 + row.icd_17_3, axis=1)
    data['icd_18_1'] = data['diag_1'].swifter.apply(lambda x: 1
                                                    if 1100 > x >= 1000 else 0)
    data['icd_18_2'] = data['diag_2'].swifter.apply(lambda x: 1
                                                    if 1100 > x >= 1000 else 0)
    data['icd_18_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if 1100 > x >= 1000 else 0)
    data['icd_18'] = data.swifter.apply(lambda row: row.icd_18_1
                                        + row.icd_18_2 + row.icd_18_3, axis=1)
    data['icd_19_1'] = data['diag_1'].swifter.apply(lambda x: 1 if
                                                    x >= 2000 else 0)
    data['icd_19_2'] = data['diag_2'].swifter.apply(lambda x: 1 if
                                                    x >= 2000 else 0)
    data['icd_19_3'] = data['diag_3'].swifter.apply(lambda x: 1
                                                    if x >= 2000 else 0)
    data['icd_19'] = data.swifter.apply(lambda row: row.icd_19_1
                                        + row.icd_19_2 + row.icd_19_3, axis=1)
    data = data.drop(columns=['icd_15_1', 'icd_15_2', 'icd_15_3',
                              'icd_16_1', 'icd_16_2', 'icd_16_3',
                              'icd_17_1', 'icd_17_2', 'icd_17_3',
                              'icd_18_1', 'icd_18_2', 'icd_18_3',
                              'icd_19_1', 'icd_19_2', 'icd_19_3'])
    return data


def number_meds(data, col_list):
    """Creates column with number of medications"""
    print('number_meds')
    for col in col_list:
        data[f"{col}_use"] = data[col].swifter.apply(lambda x: 0
                                                     if x == 'no' else 1)
    data['num_meds'] = data.iloc[:, 64:84].sum(axis=1)
    data = data.drop(columns=['metformin_use', 'repaglinide_use',
                              'nateglinide_use', 'chlorpropamide_use',
                              'glimepiride_use', 'acetohexamide_use',
                              'glipizide_use', 'glyburide_use',
                              'tolbutamide_use', 'pioglitazone_use',
                              'rosiglitazone_use', 'acarbose_use',
                              'miglitol_use', 'troglitazone_use',
                              'tolazamide_use', 'insulin_use',
                              'glyburide-metformin_use',
                              'glipizide-metformin_use',
                              'glimepiride-pioglitazone_use',
                              'metformin-pioglitazone_use'])
    return data


def num_down(data, col_list):
    """Creates column with number of medications decreased"""
    print('num_down')
    for col in col_list:
        data[f"{col}_down"] = data[col].swifter.apply(lambda x: 1
                                                      if x == 'down' else 0)
    data['num_down'] = data.iloc[:, 65:].sum(axis=1)
    data = data.drop(columns=['metformin_down', 'repaglinide_down',
                              'nateglinide_down', 'chlorpropamide_down',
                              'glimepiride_down', 'acetohexamide_down',
                              'glipizide_down', 'glyburide_down',
                              'tolbutamide_down', 'pioglitazone_down',
                              'rosiglitazone_down', 'acarbose_down',
                              'miglitol_down', 'troglitazone_down',
                              'tolazamide_down', 'insulin_down',
                              'glyburide-metformin_down',
                              'glipizide-metformin_down',
                              'glimepiride-pioglitazone_down',
                              'metformin-pioglitazone_down'])
    return data


def num_up(data, col_list):
    """Creates column with number of medications increased"""
    print('num_up')
    for col in col_list:
        data[f"{col}_up"] = data[col].swifter.apply(lambda x: 1
                                                    if x == 'up' else 0)
    data['num_up'] = data.iloc[:, 66:].sum(axis=1)
    data = data.drop(columns=['metformin_up', 'repaglinide_up',
                              'nateglinide_up', 'chlorpropamide_up',
                              'glimepiride_up', 'acetohexamide_up',
                              'glipizide_up', 'glyburide_up',
                              'tolbutamide_up', 'pioglitazone_up',
                              'rosiglitazone_up', 'acarbose_up',
                              'miglitol_up', 'troglitazone_up',
                              'tolazamide_up', 'insulin_up',
                              'glyburide-metformin_up',
                              'glipizide-metformin_up',
                              'glimepiride-pioglitazone_up',
                              'metformin-pioglitazone_up'])
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
    data = data.drop(columns=['admission_source_id', 'diag_1',
                              'diag_2', 'diag_3'])
    return data


def x_dummy_variables(x_val):
    """Produces dummy variables, corrects gender column"""
    print('X_dummy_variables')
    x_val['gender'] = x_val['gender'].replace({'female': "1", 'male': "0"})
    x_val = x_val.rename(columns={'gender': 'female'})
    x_val_race = pd.get_dummies(x_val.race, drop_first=False)
    x_val = pd.concat([x_val, x_val_race], axis=1, ignore_index=False)

    x_glucose = pd.get_dummies(x_val['max_glu_serum'], drop_first=False)
    x_val = pd.concat([x_val, x_glucose], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'200_to_300': 'glu_200_300',
                                  'more_than_300': 'glu_over_300',
                                  'norm': 'glu_norm'})
    x_val = x_val.drop(columns=['none', 'max_glu_serum', 'race',
                                'caucasian'])

    x_a1 = pd.get_dummies(x_val['a1cresult'], drop_first=False)
    x_val = pd.concat([x_val, x_a1], axis=1, ignore_index=False)

    x_val = x_val.rename(columns={'7_to_8': 'a1_7_8',
                                  'over_8': 'al_over_8', 'norm': 'a1_norm'})

    x_val = x_val.drop(columns=['nateglinide', 'chlorpropamide',
                                'acetohexamide', 'tolbutamide',
                                'acarbose', 'miglitol', 'troglitazone',
                                'tolazamide', 'glyburide-metformin',
                                'glipizide-metformin',
                                'glimepiride-pioglitazone',
                                'metformin-pioglitazone', 'none',
                                'a1cresult'])

    x_metformin = pd.get_dummies(x_val['metformin'], drop_first=False)
    x_val = pd.concat([x_val, x_metformin], axis=1, ignore_index=False)

    x_val = x_val.rename(columns={'down': 'metformin_down',
                                  'steady': 'metformin_steady',
                                  'up': 'metformin_up'})
    x_val = x_val.drop(columns=['no', 'metformin'])

    x_repaglinide = pd.get_dummies(x_val['repaglinide'], drop_first=False)
    x_val = pd.concat([x_val, x_repaglinide], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'repaglinide_down',
                                  'steady': 'repaglinide_steady',
                                  'up': 'repaglinide_up'})
    x_val = x_val.drop(columns=['no', 'repaglinide'])

    x_glimepiride = pd.get_dummies(x_val['glimepiride'], drop_first=False)
    x_val = pd.concat([x_val, x_glimepiride], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'glimepiride_down',
                                  'steady': 'glimepiride_steady',
                                  'up': 'glimepiride_up'})
    x_val = x_val.drop(columns=['no', 'glimepiride'])

    x_glipizide = pd.get_dummies(x_val['glipizide'], drop_first=False)
    x_val = pd.concat([x_val, x_glipizide], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'glipizide_down',
                                  'steady': 'glipizide_steady',
                                  'up': 'glipizide_up'})
    x_val = x_val.drop(columns=['no', 'glipizide'])

    x_glyburide = pd.get_dummies(x_val['glyburide'], drop_first=False)
    x_val = pd.concat([x_val, x_glyburide], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'glyburide_down',
                                  'steady': 'glyburide_steady',
                                  'up': 'glyburide_up'})
    x_val = x_val.drop(columns=['no', 'glyburide'])

    x_pioglitazone = pd.get_dummies(x_val['pioglitazone'], drop_first=False)
    x_val = pd.concat([x_val, x_pioglitazone], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'pioglitazone_down',
                                  'steady': 'pioglitazone_steady',
                                  'up': 'pioglitazone_up'})
    x_val = x_val.drop(columns=['no', 'pioglitazone'])

    x_rosiglitazone = pd.get_dummies(x_val['rosiglitazone'], drop_first=False)
    x_val = pd.concat([x_val, x_rosiglitazone], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'rosiglitazone_down',
                                  'steady': 'rosiglitazone_steady',
                                  'up': 'rosiglitazone_up'})
    x_val = x_val.drop(columns=['no', 'rosiglitazone'])

    x_insulin = pd.get_dummies(x_val['insulin'], drop_first=False)
    x_val = pd.concat([x_val, x_insulin], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'insulin_down',
                                  'steady': 'insulin_steady',
                                  'up': 'insulin_up'})
    x_val = x_val.drop(columns=['no', 'insulin'])

    x_val['change'] = x_val.change.replace({'ch': 1, 'no': 0})
    x_val['diabetesmed'] = x_val.diabetesmed.replace({'yes': 1, 'no': 0})

    x_val = x_val.astype('int64')
    x_val['patient_nbr'] = x_val['patient_nbr'].astype('str')
    return x_val


def clean_1():
    """Runs all cleaning functions 1_2"""
    x_1, y_1 = upload()
    y_2 = y_clean(y_1)
    x_2 = column_drop(x_1)
    x_3 = null_value_drop(x_2)
    x_4 = x_clean(x_3)
    x_5 = values_lower(x_4)
    x_6 = column_lowercase(x_5)
    x_7 = categorize_all(x_6)
    x_8 = diagnoses_1_3(x_7)
    x_9 = diagnoses_2_3(x_8)
    x_10 = diagnoses_3_3(x_9)
    x_11 = med_columns(x_10)
    x_12, y_final = reset_indices(x_11, y_2)
    return x_12, y_final


def clean():
    """Runs all cleaning functions"""
    x_12, y_final = clean_1()
    x_final = x_dummy_variables(x_12)
    x_final_no_nbr = x_final.drop(columns=['patient_nbr'])
    return x_final, y_final, x_final_no_nbr
