"""
This file contains functions for use in data preparation.
"""


import pandas as pd
import swifter
import functions as f

def upload():
    """Uploads the necessary data and returns the y
    and X databases."""
    print('Upload')
    data = pd.read_csv('data/diabetic_data.csv')
    data = data.set_index('encounter_id')
    data = data.sort_index()
    x_val = data.drop(columns=['readmitted'])
    y_val = data.readmitted
    return x_val, y_val


def y_clean(y_val):
    """Replaces readmitted data with cleaned values"""
    print('y_clean')
    y_val = y_val.replace({'NO': 0, '>30': 1, '<30': 2})
    return y_val


def null_value_drop(x_val):
    """Drops rows will null data where it will not lead
    to excessive data loss"""
    print('null_value_drop')
    x_val = x_val.loc[~(x_val.race == '?')]
    x_val = x_val.loc[~(x_val.gender == 'Unknown/Invalid')]
    x_val = x_val.loc[~((x_val.diag_1 == '?') &
                        (x_val.diag_2 == '?') & (x_val.diag_3 == '?'))]
    return x_val


def drop_invalid_data(x_val):
    """Drops invalid data"""
    #Remove individual patients' repeated visits
    x_val = x_val.loc[~(x_val.duplicated(subset=['patient_nbr']))]
    #Remove 'Newborn' patients (admission type ID = 4)
    x_val = x_val.loc[~(x_val['admission_type_id']==4)]
    #Remove incarcerated patients (admission source ID = 8)
    x_val = x_val.loc[~(x_val['admission_source_id'] == 8)]
    #Remove patients dead or discharged to hospice
    discharge_list = [11, 13, 14, 19, 20, 21]
    x_val = x_val.loc[~(x_val.discharge_disposition_id.isin(discharge_list))]
    return x_val


def reset_indices(x_val, y_val):
    """Ensures that dropped rows are reflected in y dataset.
    Resets indices for X and y datasets"""
    print('reset_indices')
    data = x_val.merge(y_val, how = 'left', left_index=True, right_index=True)
    data = data.reset_index(drop=True)
    x_val = data.drop(columns=['readmitted'])
    y_val = data.readmitted
    return x_val, y_val


def column_drop(x_val):
    """Drops columns that can't provide information due to
    either missing data or lack of variance"""
    print('column_drop')
    x_val = x_val.drop(columns=['weight', 'payer_code', 'medical_specialty',
                                'examide', 'citoglipton', 'patient_nbr',
                                'metformin-rosiglitazone'])
    return x_val


def x_clean(x_val):
    """Cleans string values in X database"""
    print('x_clean')
    x_val['race'] = x_val.race.replace({'AfricanAmerican': 'race_black',
                                        'Asian': 'race_other', 'Hispanic': 'race_other',
                                        'Other': 'race_other'})
    x_val['max_glu_serum'] = x_val.max_glu_serum.replace({'None': 0, 'Norm': 1,
                                                          '>200': 2, '>300': 3})
    x_val['age'] = x_val.age.replace({'[0-10)': 0, '[10-20)': 0,
                                      '[20-30)': 0, '[30-40)': 1,
                                      '[40-50)': 2, '[50-60)': 3,
                                      '[60-70)': 4, '[70-80)': 5,
                                      '[80-90)': 6, '[90-100)': 7})
    x_val['A1Cresult'] = x_val.A1Cresult.replace({'None': 0, 'Norm': 1, '>7': 2,
                                                  ">8": 3})
    x_val = x_val.rename(columns = {'admission_type_id': 'at_id',
                                    'discharge_disposition_id': 'dd_id',
                                    'admission_source_id': 'as_id',
                                    'number_outpatient': 'outpatient',
                                    'number_emergency': 'emergency',
                                    'number_inpatient': 'inpatient'})
    x_val['at_id'] = x_val.at_id.replace({1: 'at_urgent', 2: 'at_urgent', 3: 'at_not_urgent',
                                          5: 'at_no_info', 6: 'at_no_info', 7: 'at_urgent',
                                          8: 'at_no_info'})
    x_val['dd_id'] = x_val['dd_id'].replace({1: 'dd_home', 2: 'dd_care', 3: 'dd_care', 4: 'dd_care',
                                             5: 'dd_care', 6: 'dd_care_home', 7: 'dd_other',
                                             8: 'dd_care_home', 9: 'dd_care', 10: 'dd_care',
                                             12: 'dd_care', 15: 'dd_care', 16: 'dd_care',
                                             17: 'dd_care', 18: 'no_info', 22: 'dd_care_lt',
                                             23: 'dd_care_lt', 24: 'dd_care_lt', 25: 'dd_no_info',
                                             27: 'dd_care_lt', 28: 'dd_care_lt'})
    x_val['as_id'] = x_val['as_id'].replace({1:'as_referral', 2: 'as_referral', 3: 'as_referral',
                                             4: 'as_transfer_other', 5: 'as_transfer_other',
                                             6: 'as_transfer_other', 7: 'as_transfer_er',
                                             9: 'as_no_info', 10: 'as_transfer_other',
                                             11: 'as_transfer_other', 12: 'as_transfer_other',
                                             13: 'as_transfer_other', 14: 'as_transfer_other',
                                             15: 'as_no_info', 17: 'as_no_info', 20: 'as_no_info',
                                             22: 'as_transfer_other', 25: 'as_transfer_other'})
    return x_val


def values_lower(x_val):
    """Makes all string values lowercase"""
    print('values_lower')
    x_val_str_cols = list((x_val.select_dtypes(include=['object'])).columns)
    for col in x_val_str_cols:
        x_val[col] = x_val[col].apply(lambda xii: xii.lower())
    return x_val


def column_lowercase(x_val):
    """Makes all column names lowercase"""
    print('column_lowercase')
    x_val_cols = list(x_val.columns)
    for col in x_val_cols:
        x_val = x_val.rename(columns={str(col): col.lower()})
    return x_val


def categorize_all(x_val):
    """Transforms diagnosis data for all three rows"""
    print('categorize_all')
    x_val['diag_1'] = x_val.diag_1.swifter.apply(lambda x: float(x[1:])+2000
                                                 if x[:1] == 'e'
                                                 else (float(x[1:])+1000
                                                       if x[:1] == 'v'
                                                       else (0 if x[:1] == "?"
                                                             else (float(x)))))
    x_val['diag_2'] = x_val.diag_2.swifter.apply(lambda x: float(x[1:])+2000
                                                 if x[:1] == 'e'
                                                 else (float(x[1:])+1000
                                                       if x[:1] == 'v'
                                                       else (0 if x[:1] == "?"
                                                             else (float(x)))))
    x_val['diag_3'] = x_val.diag_3.swifter.apply(lambda x: float(x[1:])+2000
                                                 if x[:1] == 'e'
                                                 else (float(x[1:])+1000
                                                       if x[:1] == 'v'
                                                       else (0 if x[:1] == "?"
                                                             else (float(x)))))
    return x_val


def diagnoses_1_3(x_val):
    """Creates a column summing the number of diagnoses from each category"""
    print('diagnoses_1_3')
    x_val['icd_1_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 140 > x >= 1 else 0)
    x_val['icd_1_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 140 > x >= 1 else 0)
    x_val['icd_1_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 140 > x >= 1 else 0)
    x_val['icd_1'] = x_val.swifter.apply(lambda row: row.icd_1_1
                                         + row.icd_1_2 + row.icd_1_3, axis=1)
    x_val['icd_2_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 240 > x >= 140 else 0)
    x_val['icd_2_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 240 > x >= 140 else 0)
    x_val['icd_2_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 240 > x >= 140 else 0)
    x_val['icd_2'] = x_val.swifter.apply(lambda row: row.icd_2_1 + row.icd_2_2
                                         + row.icd_2_3, axis=1)
    x_val['icd_3_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 280 > x >= 240 else 0)
    x_val['icd_3_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 280 > x >= 240 else 0)
    x_val['icd_3_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 280 > x >= 240 else 0)
    x_val['icd_3'] = x_val.swifter.apply(lambda row: row.icd_3_1 + row.icd_3_2
                                         + row.icd_3_3, axis=1)
    x_val['icd_4_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 290 > x >= 280 else 0)
    x_val['icd_4_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 290 > x >= 280 else 0)
    x_val['icd_4_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 290 > x >= 280 else 0)
    x_val['icd_4'] = x_val.swifter.apply(lambda row: row.icd_4_1 + row.icd_4_2
                                         + row.icd_4_3, axis=1)
    x_val['icd_5_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 320 > x >= 290 else 0)
    x_val['icd_5_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 320 > x >= 290 else 0)
    x_val['icd_5_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 320 > x >= 290 else 0)
    x_val['icd_5'] = x_val.swifter.apply(lambda row: row.icd_5_1 + row.icd_5_2
                                         + row.icd_5_3, axis=1)
    x_val['icd_6_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 390 > x >= 320 else 0)
    x_val['icd_6_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 390 > x >= 320 else 0)
    x_val['icd_6_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 390 > x >= 320 else 0)
    x_val['icd_6'] = x_val.swifter.apply(lambda row: row.icd_6_1 + row.icd_6_2
                                         + row.icd_6_3, axis=1)
    x_val['icd_7_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 460 > x >= 390 else 0)
    x_val['icd_7_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 460 > x >= 390 else 0)
    x_val['icd_7_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 460 > x >= 390 else 0)
    x_val['icd_7'] = x_val.swifter.apply(lambda row: row.icd_7_1
                                         + row.icd_7_2 + row.icd_7_3, axis=1)
    x_val = x_val.drop(columns=['icd_1_1', 'icd_1_2', 'icd_1_3',
                                'icd_2_1', 'icd_2_2', 'icd_2_3',
                                'icd_3_1', 'icd_3_2', 'icd_3_3',
                                'icd_4_1', 'icd_4_2', 'icd_4_3',
                                'icd_5_1', 'icd_5_2', 'icd_5_3',
                                'icd_6_1', 'icd_6_2', 'icd_6_3',
                                'icd_7_1', 'icd_7_2', 'icd_7_3'])
    return x_val


def diagnoses_2_3(x_val):
    """Creates a column summing the number of diagnoses from each category"""
    print('diagnoses_2_3')
    x_val['icd_8_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 520 > x >= 460 else 0)
    x_val['icd_8_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 520 > x >= 460 else 0)
    x_val['icd_8_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 520 > x >= 460 else 0)
    x_val['icd_8'] = x_val.swifter.apply(lambda row: row.icd_8_1
                                         + row.icd_8_2 + row.icd_8_3, axis=1)
    x_val['icd_9_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                     if 580 > x >= 520 else 0)
    x_val['icd_9_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                     if 580 > x >= 520 else 0)
    x_val['icd_9_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                     if 580 > x >= 520 else 0)
    x_val['icd_9'] = x_val.swifter.apply(lambda row: row.icd_9_1
                                         + row.icd_9_2 + row.icd_9_3, axis=1)
    x_val['icd_10_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 630 > x >= 580 else 0)
    x_val['icd_10_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 630 > x >= 580 else 0)
    x_val['icd_10_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 630 > x >= 580 else 0)
    x_val['icd_10'] = x_val.swifter.apply(lambda row: row.icd_10_1
                                          + row.icd_10_2 + row.icd_10_3, axis=1)
    x_val['icd_11_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 680 > x >= 630 else 0)
    x_val['icd_11_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 680 > x >= 630 else 0)
    x_val['icd_11_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 680 > x >= 630 else 0)
    x_val['icd_11'] = x_val.swifter.apply(lambda row: row.icd_11_1
                                          + row.icd_11_2 + row.icd_11_3, axis=1)
    x_val['icd_12_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 710 > x >= 680 else 0)
    x_val['icd_12_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 710 > x >= 680 else 0)
    x_val['icd_12_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 710 > x >= 680 else 0)
    x_val['icd_12'] = x_val.swifter.apply(lambda row: row.icd_12_1
                                          + row.icd_12_2 + row.icd_12_3, axis=1)
    x_val['icd_13_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 740 > x >= 710 else 0)
    x_val['icd_13_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 740 > x >= 710 else 0)
    x_val['icd_13_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 740 > x >= 710 else 0)
    x_val['icd_13'] = x_val.swifter.apply(lambda row: row.icd_13_1
                                          + row.icd_13_2 + row.icd_13_3, axis=1)
    x_val['icd_14_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 760 > x >= 740 else 0)
    x_val['icd_14_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 760 > x >= 740 else 0)
    x_val['icd_14_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 760 > x >= 740 else 0)
    x_val['icd_14'] = x_val.swifter.apply(lambda row: row.icd_14_1
                                          + row.icd_14_2 + row.icd_14_3, axis=1)
    x_val = x_val.drop(columns=['icd_8_1', 'icd_8_2', 'icd_8_3', 'icd_9_1',
                                'icd_9_2', 'icd_9_3', 'icd_10_1', 'icd_10_2',
                                'icd_10_3', 'icd_11_1', 'icd_11_2', 'icd_11_3',
                                'icd_12_1', 'icd_12_2', 'icd_12_3', 'icd_13_1',
                                'icd_13_2', 'icd_13_3', 'icd_14_1', 'icd_14_2',
                                'icd_14_3'])
    return x_val


def diagnoses_3_3(x_val):
    """Creates a column summing the number of diagnoses from each category.
    Skips icd_15, because no instances exist."""
    print('diagnoses_3_3')
    x_val['icd_16_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 800 > x >= 780 else 0)
    x_val['icd_16_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 800 > x >= 780 else 0)
    x_val['icd_16_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 800 > x >= 780 else 0)
    x_val['icd_16'] = x_val.swifter.apply(lambda row: row.icd_16_1
                                          + row.icd_16_2 + row.icd_16_3, axis=1)
    x_val['icd_17_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 1000 > x >= 800 else 0)
    x_val['icd_17_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 1000 > x >= 800 else 0)
    x_val['icd_17_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 1000 > x >= 800 else 0)
    x_val['icd_17'] = x_val.swifter.apply(lambda row: row.icd_17_1 +
                                          row.icd_17_2 + row.icd_17_3, axis=1)
    x_val['icd_18_1'] = x_val['diag_1'].swifter.apply(lambda x: 1
                                                      if 1100 > x >= 1000 else 0)
    x_val['icd_18_2'] = x_val['diag_2'].swifter.apply(lambda x: 1
                                                      if 1100 > x >= 1000 else 0)
    x_val['icd_18_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if 1100 > x >= 1000 else 0)
    x_val['icd_18'] = x_val.swifter.apply(lambda row: row.icd_18_1
                                          + row.icd_18_2 + row.icd_18_3, axis=1)
    x_val['icd_19_1'] = x_val['diag_1'].swifter.apply(lambda x: 1 if
                                                      x >= 2000 else 0)
    x_val['icd_19_2'] = x_val['diag_2'].swifter.apply(lambda x: 1 if
                                                      x >= 2000 else 0)
    x_val['icd_19_3'] = x_val['diag_3'].swifter.apply(lambda x: 1
                                                      if x >= 2000 else 0)
    x_val['icd_19'] = x_val.swifter.apply(lambda row: row.icd_19_1
                                          + row.icd_19_2 + row.icd_19_3, axis=1)
    x_val = x_val.drop(columns=['icd_16_1', 'icd_16_2', 'icd_16_3',
                                'icd_17_1', 'icd_17_2', 'icd_17_3',
                                'icd_18_1', 'icd_18_2', 'icd_18_3',
                                'icd_19_1', 'icd_19_2', 'icd_19_3'])
    return x_val


def number_meds(x_val, col_list):
    """Creates column with number of medications"""
    print('number_meds')
    for col in col_list:
        x_val[f"{col}_use"] = x_val[col].swifter.apply(lambda x: 0
                                                       if x == 'no' else 1)
    x_val['num_meds'] = x_val.iloc[:, 64:84].sum(axis=1)
    x_val = x_val.drop(columns=['metformin_use', 'repaglinide_use',
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
    return x_val


def num_down(x_val, col_list):
    """Creates column with number of medications decreased"""
    print('num_down')
    for col in col_list:
        x_val[f"{col}_down"] = x_val[col].swifter.apply(lambda x: 1
                                                        if x == 'down' else 0)
    x_val['num_down'] = x_val.iloc[:, 65:].sum(axis=1)
    x_val = x_val.drop(columns=['metformin_down', 'repaglinide_down',
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
    return x_val


def num_up(x_val, col_list):
    """Creates column with number of medications increased"""
    print('num_up')
    for col in col_list:
        x_val[f"{col}_up"] = x_val[col].swifter.apply(lambda x: 1
                                                      if x == 'up' else 0)
    x_val['num_up'] = x_val.iloc[:, 66:].sum(axis=1)
    x_val = x_val.drop(columns=['metformin_up', 'repaglinide_up',
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
    return x_val


def med_columns(x_val):
    """Creates all new columns related to medication quantity and
    change"""
    col_list = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-pioglitazone']
    x_val = number_meds(x_val, col_list)
    x_val = num_down(x_val, col_list)
    x_val = num_up(x_val, col_list)
    x_val = x_val.drop(columns=['diag_1', 'diag_2', 'diag_3'])
    return x_val


def x_dummy_variables(x_val):
    """Produces dummy variables, corrects gender column"""
    print('X_dummy_variables')
    x_val['gender'] = x_val['gender'].replace({'female': "1", 'male': "0"})
    x_val = x_val.rename(columns={'gender': 'female'})
    x_val_race = pd.get_dummies(x_val.race, drop_first=False)
    x_val = pd.concat([x_val, x_val_race], axis=1, ignore_index=False)

    x_val = x_val.drop(columns=['nateglinide', 'chlorpropamide',
                                'acetohexamide', 'tolbutamide',
                                'acarbose', 'miglitol', 'troglitazone',
                                'tolazamide', 'glyburide-metformin',
                                'glipizide-metformin',
                                'glimepiride-pioglitazone',
                                'metformin-pioglitazone', 'caucasian', 'race'])

    x_at = pd.get_dummies(x_val['at_id'], drop_first=False)
    x_val = pd.concat([x_val, x_at], axis=1, ignore_index=False)
    x_val = x_val.drop(columns=['at_id', 'at_no_info'])

    x_dd = pd.get_dummies(x_val['dd_id'], drop_first=False)
    x_val = pd.concat([x_val, x_dd], axis=1, ignore_index=False)
    x_val = x_val.drop(columns=['dd_id', 'dd_no_info'])

    x_as = pd.get_dummies(x_val['as_id'], drop_first=False)
    x_val = pd.concat([x_val, x_as], axis=1, ignore_index=False)
    x_val = x_val.drop(columns=['as_id', 'as_no_info'])

    x_metformin = pd.get_dummies(x_val['metformin'], drop_first=False)
    x_val = pd.concat([x_val, x_metformin], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'metformin_down', 'steady': 'metformin_steady',
                                  'up': 'metformin_up'})
    x_val = x_val.drop(columns=['no', 'metformin'])

    x_repaglinide = pd.get_dummies(x_val['repaglinide'], drop_first=False)
    x_val = pd.concat([x_val, x_repaglinide], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'repaglinide_down', 'steady': 'repaglinide_steady',
                                  'up': 'repaglinide_up'})
    x_val = x_val.drop(columns=['no', 'repaglinide'])

    x_glimepiride = pd.get_dummies(x_val['glimepiride'], drop_first=False)
    x_val = pd.concat([x_val, x_glimepiride], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'glimepiride_down', 'steady': 'glimepiride_steady',
                                  'up': 'glimepiride_up'})
    x_val = x_val.drop(columns=['no', 'glimepiride'])

    x_glipizide = pd.get_dummies(x_val['glipizide'], drop_first=False)
    x_val = pd.concat([x_val, x_glipizide], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'glipizide_down', 'steady': 'glipizide_steady',
                                  'up': 'glipizide_up'})
    x_val = x_val.drop(columns=['no', 'glipizide'])

    x_glyburide = pd.get_dummies(x_val['glyburide'], drop_first=False)
    x_val = pd.concat([x_val, x_glyburide], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'glyburide_down', 'steady': 'glyburide_steady',
                                  'up': 'glyburide_up'})
    x_val = x_val.drop(columns=['no', 'glyburide'])

    x_pioglitazone = pd.get_dummies(x_val['pioglitazone'], drop_first=False)
    x_val = pd.concat([x_val, x_pioglitazone], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'pioglitazone_down', 'steady': 'pioglitazone_steady',
                                  'up': 'pioglitazone_up'})
    x_val = x_val.drop(columns=['no', 'pioglitazone'])

    x_rosiglitazone = pd.get_dummies(x_val['rosiglitazone'], drop_first=False)
    x_val = pd.concat([x_val, x_rosiglitazone], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'rosiglitazone_down', 'steady': 'rosiglitazone_steady',
                                  'up': 'rosiglitazone_up'})
    x_val = x_val.drop(columns=['no', 'rosiglitazone'])

    x_insulin = pd.get_dummies(x_val['insulin'], drop_first=False)
    x_val = pd.concat([x_val, x_insulin], axis=1, ignore_index=False)
    x_val = x_val.rename(columns={'down': 'insulin_down', 'steady': 'insulin_steady',
                                  'up': 'insulin_up'})
    x_val = x_val.drop(columns=['no', 'insulin'])

    x_val['change'] = x_val.change.replace({'ch': 1, 'no': 0})
    x_val['diabetesmed'] = x_val.diabetesmed.replace({'yes': 1, 'no': 0})
    x_val = x_val.astype('int64')
    return x_val


def outliers_bin(x_val):
    """Bins some continuous features with significant outliers"""
    binned_columns = ['outpatient', 'emergency', 'inpatient']
    for col in binned_columns:
        x_val[f'{col}_binned'] = x_val[col].swifter.apply(lambda x: 3 if x>3 else
                                                          (2 if 2<=x<=3 else x))
    x_val = x_val.drop(columns = binned_columns)
    return x_val


def outliers_log(x_val):
    """Creates log of some continuous features with significant outliers"""
    log_columns = ['time_in_hospital', 'num_lab_procedures', 'num_medications']
    for column in log_columns:
        x_val[f'{column}_log'] = np.log(x_val[column])
    x_val = x_val.drop(columns = log_columns)


def clean():
    """Runs all cleaning functions"""
    x_1, y_1 = upload()
    y_2 = y_clean(y_1)
    x_2 = null_value_drop(x_1)
    x_3 = drop_invalid_data(x_2)
    x_4, y_final = reset_indices(x_3, y_2)
    x_5 = column_drop(x_4)
    x_6 = x_clean(x_5)
    x_7 = values_lower(x_6)
    x_8 = column_lowercase(x_7)
    x_9 = categorize_all(x_8)
    x_10 = diagnoses_1_3(x_9)
    x_11 = diagnoses_2_3(x_10)
    x_12 = diagnoses_3_3(x_11)
    x_13 = med_columns(x_12)
    x_14 = x_dummy_variables(x_13)
    x_15 = outliers_bin(x_14)
    x_final = outliers_log(x_15)
    return x_final, y_final
