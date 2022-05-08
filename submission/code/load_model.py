import numpy as np
import xgboost as xgb
import os
import json
from joblib import load
from utils import load_as_dataframe, get_filenames
from stats import extract_features


def boost_predict(model_folder, df):
    bst = xgb.Booster()
    bst.load_model(os.path.join(model_folder, "model.json"))

    with open(os.path.join(model_folder, 'hyperparams.json'), 'r') as f:
        hyperparams = json.load(f)

    with open(os.path.join(model_folder, 'train_params.json'), 'r') as f:
        train_params = json.load(f)

    features = extract_features([df], chunk_size=hyperparams['chunk_size'], n_pca=hyperparams['n_pca']).to_numpy()

    dtest = xgb.DMatrix(data=features.T, label=np.zeros(features.shape[1]))

    return np.clip(np.round(bst.predict(dtest, ntree_limit=bst.best_ntree_limit)), 0, 1)


def svm_predict(model_folder, df):
    clf = load(os.path.join(model_folder, 'model.joblib'))

    with open(os.path.join(model_folder, 'hyperparams.json'), 'r') as f:
        hyperparams = json.load(f)

    features = extract_features([df], chunk_size=hyperparams['chunk_size'], n_pca=hyperparams['n_pca']).to_numpy().T

    pred = clf.predict(features)

    return pred


if __name__ == '__main__':
    model_folder = 'svm_acc_95_2022_05_08_16_19_22'

    folder = 'data'
    extension = '.npz'
    subfolder = 'mixed'
    filename = get_filenames(folder, subfolder=subfolder, extension=extension, max_files=1, shuffle=True)
    df = load_as_dataframe(os.path.join(folder, subfolder), filename[0])

    print(filename)
    # pred = boost_predict(model_folder, df)
    pred = svm_predict(model_folder, df)
