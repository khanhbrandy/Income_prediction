"""
Created on 2020-01-25
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

class Preprocessor:
    def __init__(self):
        self.impute_etc = ExtraTreeClassifier()
        self.impute_dtc = DecisionTreeClassifier()
        self.impute_rfc = RandomForestClassifier()

    def get_data(self, url):
        columns = ['age',
        'class of worker',
        'detailed industry recode',
        'detailed occupation recode',
        'education',
        'wage per hour',
        'enroll in edu inst last wk',
        'marital status',
        'major industry code',
        'major occupation code',
        'race',
        'hispanic origin',
        'sex',
        'member of a labor union',
        'reason for unemployment',
        'full or part time employment stat',
        'capital gains',
        'capital losses',
        'dividends from stocks',
        'tax filer stat',
        'region of previous residence',
        'state of previous residence',
        'detailed household and family stat',
        'detailed household summary in household',
        'instance weight',
        'migration code-change in msa',
        'migration code-change in reg',
        'migration code-move within reg',
        'live in this house 1 year ago',
        'migration prev res in sunbelt',
        'num persons worked for employer',
        'family members under 18',
        'country of birth father',
        'country of birth mother',
        'country of birth self',
        'citizenship',
        'own business or self employed',
        'fill inc questionnaire for veterans admin',
        'veterans benefits',
        'weeks worked in year',
        'year',
        'class'
        ]
        data=pd.read_csv(url, names = columns, na_values=' ?')
        for col in data.select_dtypes('O').columns:
            data[col] = data[col].astype('category')
        return data

    def get_null(self, data):
        contain_null = np.array(data.isnull().sum().to_frame()[data.isnull().sum().to_frame()[0]!=0].index)
        return contain_null

    def OnehotEncode(self, data, categorical_columns):
        df_1 = data.drop(columns = categorical_columns, axis = 1)
        df_2 = pd.get_dummies(data[categorical_columns])
        df = pd.concat([df_1, df_2], axis=1, join='inner')
        return df

    def ImputeVoteClassifier(self, data, target_name):
        print('*'*100+'\n')
        print('Start imputing missing values for feature: {} \n'.format(target_name))
        # Training set
        print('Generating training set...')
        train_data = data[data[target_name].notnull()].copy()
        train_target = train_data[target_name]
        train_data.drop(columns = [target_name], inplace = True)
        encoded_train = OnehotEncode(train_data, train_data.select_dtypes('category').columns)
        print('Done generating training set \n')
        # Testing set
        print('Generating testing set...')
        test_data = data[data[target_name].isnull()].copy()
        test_target = test_data[target_name]
        # Drop target var in testing set
        test_data.drop(columns = [target_name], inplace = True)
        encoded_test = self.OnehotEncode(test_data, test_data.select_dtypes('category').columns)
        print('Done generating testing set \n')
        # Fit data into base classifiers
        etc = self.ExtraTreeClassifier()
        print('Fitting data into {}...'.format(etc.__class__.__name__))
        etc.fit(encoded_train, train_target)
        etc_pred = etc.predict(encoded_test)

        dtc = self.DecisionTreeClassifier()
        print('Fitting data into {}...'.format(dtc.__class__.__name__))
        dtc.fit(encoded_train, train_target)
        dtc_pred = dtc.predict(encoded_test)

        rfc = self.RandomForestClassifier()
        print('Start fitting data into {}...'.format(rfc.__class__.__name__))
        rfc.fit(encoded_train, train_target)
        rfc_pred = rfc.predict(encoded_test)
        
        # Finalize data
        print('Voting final predictions...')
        final_pred = np.array([])
        for i in range(0,len(test_target)):
            final_pred = np.append(final_pred, mode([etc_pred[i], dtc_pred[i], rfc_pred[i]])[0])
        print('Done voting and dump final predictions into feature: {}'.format(target_name))
        print('\n'+'*'*100)
        return final_pred



