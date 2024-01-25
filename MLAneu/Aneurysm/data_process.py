import pandas as pd
import numpy as np
import math

loc = {
    "A1 segment ant": 1,
    "A2 segment ant": 2,
    "AICA": 3,
    "Ant Choroidal segment carotid": 4,
    "Ant communicating artery": 5,
    "Anterior and superior wall carotid": 6,
    "Basilar Tip": 7,
    "Basilar trunk": 8,
    "Carotid bifurcation": 9,
    "Distal ant cerebral artery": 10,
    "Distal posterior cerebral artery": 11,
    "Distal to sylvian bifurcation": 12,
    "Intracavernous internal carotid": 13,
    "M1 segment middle cerebral artery": 14,
    "Medial wall carotid": 15,
    "Ophthalmic artery": 16,
    "Ophthalmic segment carotid": 17,
    "P1 Posterior cerebral artery": 18,
    "P1-P2 junction posterior cerebral artery": 19,
    "P2 posterior cerebral artery": 20,
    "Pericallosal cerebral artery": 21,
    "PICA": 22,
    "Posterior Comm": 23,
    "Superior cerebellar artery": 24,
    "Sylvian bifurcation": 25,
    "V4 segment vertebral artery": 26}

side = {
    'Midline': 1,
    'Left': 2,
    'Right': 3}


def fd_pass_process(seed=42):
    '''
    Process FD-PASS data and generate training and test sets
    '''
    data_path = "..\data\FD-PASS_data_NatComm.xlsx"
    df1 = pd.read_excel(data_path, sheet_name='Results')
    all_data = df1.values
    features = all_data[5:, :13]
    features[:, 1] = (features[:, 1] == 'Right') + 0
    features[:, 2] = (features[:, 2] - np.mean(features[:, 2], axis=0)) / np.std(features[:, 2], axis=0)
    features[:, 4:] = (features[:, 4:] - np.mean(features[:, 4:], axis=0)) / \
                      np.std(features[:, 4:].astype(np.float32), axis=0)
    hyper_label = (all_data[5:, -12:] > 35).all(axis=1)
    norm_label = (all_data[5:, -24:-12] > 35).all(axis=1)

    shuffled_id = np.arange(features.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffled_id)
    features = features[shuffled_id, :]
    hyper_label = hyper_label[shuffled_id]
    norm_label = norm_label[shuffled_id]

    train_id = math.ceil(norm_label.shape[0] * 6 / 10)
    valid_id = math.ceil(norm_label.shape[0] * 8 / 10)

    train_features = features[:train_id, :]
    valid_features = features[train_id:valid_id, :]
    test_features = features[valid_id:, :]

    train_nlabel = norm_label[:train_id] + 0
    train_hlabel = hyper_label[:train_id] + 0
    valid_nlabel = norm_label[train_id:valid_id] + 0
    valid_hlabel = hyper_label[train_id:valid_id] + 0
    test_nlabel = norm_label[valid_id:] + 0
    test_hlabel = hyper_label[valid_id:] + 0

    return train_features, valid_features, test_features, train_nlabel, train_hlabel,\
        valid_nlabel, valid_hlabel, test_nlabel, test_hlabel


def aneurist_process(seed=42):
    '''
        Process AneurIST data and generate training and test sets
    '''
    data_path = "..\data\AneurIST_Database.xlsx"
    df1 = pd.read_excel(data_path, sheet_name='database')
    features = df1[df1.columns[:8]]
    features = features[features['Ruptured'].isin(['Y', 'N', 'unruptured', 'ruptured'])]
    label = features['Ruptured'].values
    label = np.where(np.logical_or(label == 'Y', label == 'ruptured'), 1, 0)
    features = features.values
    features = np.delete(features, 5, axis=1)
    age_ave = np.ceil(np.nanmean(features[:, 6]))
    for i in range(features.shape[0]):
        if features[i, 2] in loc.keys():
            features[i, 2] = loc[features[i, 2]]
        else:
            features[i, 2] = 0
        if features[i, 3] in loc.keys():
            features[i, 3] = loc[features[i, 3]]
        else:
            features[i, 3] = 0

        if features[i, 4] in side.keys():
            features[i, 4] = side[features[i, 4]]
        else:
            features[i, 4] = 0

        if features[i, 5] == 'male':
            features[i, 5] = 1
        elif features[i, 5] == 'female':
            features[i, 5] = 2
        else:
            features[i, 5] = 0

        if np.isnan(features[i, 6]):
            features[i, 6] = age_ave

    shuffled_id = np.arange(features.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffled_id)
    features = features[shuffled_id, :]
    label = label[shuffled_id]

    features[:, 1:] = (features[:, 1:] - np.mean(features[:, 1:], axis=0)) / \
                      np.std(features[:, 1:].astype(np.float32), axis=0)

    train_id = math.ceil(label.shape[0] * 6 / 10)
    valid_id = math.ceil(label.shape[0] * 8 / 10)
    train_features = features[:train_id, :]
    valid_features = features[train_id:valid_id, :]
    test_features = features[valid_id:, :]
    train_label = label[:train_id]
    valid_label = label[train_id:valid_id]
    test_label = label[valid_id:]

    return train_features, valid_features, test_features, train_label, valid_label, test_label
