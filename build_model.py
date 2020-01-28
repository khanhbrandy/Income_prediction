"""
Created on 2019-12-26
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
import preprocess
import model
import joblib
import warnings

def preprocess_data(url, seed):
    preprocessor = preprocess.Preprocessor()
    raw_data = preprocessor.get_data(url)
    contain_null = preprocessor.get_null(raw_data)
    for f in contain_null:
        raw_data.loc[(raw_data[f].isnull()),f] = preprocessor.ImputeVoteClassifier(raw_data, f)
    X_train, y_train, X_test, y_test = preprocessor.split_data(raw_data, seed, re=False)
    return X_train, y_train, X_test, y_test
def build_model(X_train, y_train, X_test, y_test, n_fold, seed):
    modeler = model.Model()
    clf_list = [modeler.clf_0, modeler.clf_2]
    oof_train = []
    oof_test = []
    for clf in clf_list:
        clf_oof_train, clf_oof_test = modeler.generate_oof(clf, X_train, y_train, X_test, n_fold, seed)
        oof_train.append(pd.DataFrame(clf_oof_train))
        oof_test.append(pd.DataFrame(clf_oof_test))
    meta_train = modeler.generate_metadata(oof_train)
    meta_test = modeler.generate_metadata(oof_test)
    # Fit Meta classifier
    meta_clf = modeler.model_predict(modeler.clf_3, meta_train, y_train, meta_test, y_test, seed)
    print('Start dumping Meta classifier...')
    joblib.dump(meta_clf, 'meta_clf.pkl') 
    print('Done dumping Meta classifier ! \n')
    return meta_clf
if __name__=='__main__':
    print('*'*100) 
    print('*'*100+'\n')
    seed = 1003
    n_fold = 5
    colab = True
    if colab:
        import sys
        from google.colab import drive
        drive.mount('/content/drive',force_remount=True)
        sys.path.append('drive/My Drive/Colab Notebooks/Income_prediction')
        prefix = 'drive/My Drive/Colab Notebooks/Income_prediction/'
    else:
        prefix = ''
    url = prefix+'census/census-income.data'
    X_train, y_train, X_test, y_test = preprocess_data(url, seed)
    meta_clf = build_model(X_train, y_train, X_test, y_test, n_fold, seed)
    

