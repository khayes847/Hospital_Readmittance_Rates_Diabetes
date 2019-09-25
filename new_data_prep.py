import pandas as pd

def new_data_prep(X, y):
    data = pd.concat([X, y], axis=1)

    data = data.sort_index()

    #data = data.loc[data.duplicated(subset=['patient_nbr']) == False]

    discharge_list = [11, 13, 14, 19, 20, 21]

    data = data.loc[data.discharge_disposition_id.isin(discharge_list) == False]

    X = data.drop(columns = ['readmitted'])
    y = data.readmitted

    X['gender'] = X.gender.replace({'female': '1', 'male': '0'})
    X = X.rename(columns = {'gender': 'female'})

    X_race = pd.get_dummies(X.race, drop_first=False)
    X = pd.concat([X, X_race], axis=1, ignore_index=False)
    X = X.drop(columns = ['race', 'caucasian'])

    X = X.drop(columns = ['diag_1', 'diag_2', 'diag_3'])

    X_glucose = pd.get_dummies(X['max_glu_serum'], drop_first=False)
    X = pd.concat([X, X_glucose], axis=1, ignore_index=False)
    X = X.rename(columns={'200_to_300': 'glu_200_300',
                          'more_than_300': 'glu_over_300', 'norm': 'glu_norm'})
    X = X.drop(columns=['none', 'max_glu_serum'])

    X_a1 = pd.get_dummies(X['a1cresult'], drop_first=False)
    X = pd.concat([X, X_a1], axis=1, ignore_index=False)

    X = X.rename(columns={'7_to_8': 'a1_7_8',
                          'over_8': 'al_over_8', 'norm': 'a1_norm'})
    X = X.drop(columns=['none', 'a1cresult'])

    X = X.drop(columns=['nateglinide', 'chlorpropamide', 'acetohexamide',
                        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone',
                        'tolazamide', 'glyburide-metformin', 'glipizide-metformin',
                        'glimepiride-pioglitazone', 'metformin-pioglitazone'])

    X_metformin = pd.get_dummies(X['metformin'], drop_first=False)
    X = pd.concat([X, X_metformin], axis=1, ignore_index=False)

    X = X.rename(columns={'down': 'metformin_down',
                          'steady': 'metformin_steady', 'up': 'metformin_up'})
    X = X.drop(columns=['no', 'metformin'])

    X_repaglinide = pd.get_dummies(X['repaglinide'], drop_first=False)
    X = pd.concat([X, X_repaglinide], axis=1, ignore_index=False)
    X = X.rename(columns={'down': 'repaglinide_down',
                          'steady': 'repaglinide_steady', 'up': 'repaglinide_up'})
    X = X.drop(columns=['no', 'repaglinide'])

    X_glimepiride = pd.get_dummies(X['glimepiride'], drop_first=False)
    X = pd.concat([X, X_glimepiride], axis=1, ignore_index=False)
    X = X.rename(columns={'down': 'glimepiride_down',
                          'steady': 'glimepiride_steady', 'up': 'glimepiride_up'})
    X = X.drop(columns=['no', 'glimepiride'])

    X_glipizide = pd.get_dummies(X['glipizide'], drop_first=False)
    X = pd.concat([X, X_glipizide], axis=1, ignore_index=False)
    X = X.rename(columns={'down': 'glipizide_down',
                          'steady': 'glipizide_steady', 'up': 'glipizide_up'})
    X = X.drop(columns=['no', 'glipizide'])

    X_glyburide = pd.get_dummies(X['glyburide'], drop_first=False)
    X = pd.concat([X, X_glyburide], axis=1, ignore_index=False)
    X = X.rename(columns={'down': 'glyburide_down',
                          'steady': 'glyburide_steady', 'up': 'glyburide_up'})
    X = X.drop(columns=['no', 'glyburide'])

    X_pioglitazone = pd.get_dummies(X['pioglitazone'], drop_first=False)
    X = pd.concat([X, X_pioglitazone], axis=1, ignore_index=False)
    X = X.rename(columns={'down': 'pioglitazone_down',
                          'steady': 'pioglitazone_steady', 'up': 'pioglitazone_up'})
    X = X.drop(columns=['no', 'pioglitazone'])

    X_rosiglitazone = pd.get_dummies(X['rosiglitazone'], drop_first=False)
    X = pd.concat([X, X_rosiglitazone], axis=1, ignore_index=False)
    X = X.rename(columns={'down': 'rosiglitazone_down',
                          'steady': 'rosiglitazone_steady', 'up': 'rosiglitazone_up'})
    X = X.drop(columns=['no', 'rosiglitazone'])

    X_insulin = pd.get_dummies(X['insulin'], drop_first=False)
    X = pd.concat([X, X_insulin], axis=1, ignore_index=False)
    X = X.rename(columns={'down': 'insulin_down',
                          'steady': 'insulin_steady', 'up': 'insulin_up'})
    X = X.drop(columns=['no', 'insulin'])

    X['change'] = X.change.replace({'ch': 1, 'no': 0})
    X['diabetesmed'] = X.diabetesmed.replace({'yes': 1, 'no': 0})

    X['female'] = X['female'].astype('int64')

    X = X.astype('int64')

    X = X.drop(columns=[#'patient_nbr',
                        'admission_type_id',
                        'discharge_disposition_id', 'admission_source_id', ])

    y = y.replace({'no': 0, 'over_30_days': 1, 'under_30_days': 1})

    data = pd.concat([X, y], axis=1)

    return X, y
