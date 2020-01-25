"""
Created on 2019-12-26
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from sklearn import preprocessing

class Preprocessor:
    def __init__(self):
        pass

    def get_data(self, url):
        data=pd.read_csv(url, header = 0)
        return data

    def data_standardize(self, data, std=False):
        if std:
            data.set_index('FACEBOOK_ID', inplace=True)
            scaler = preprocessing.StandardScaler()
            data_stded=scaler.fit_transform(data.values)
            df_data_stded=pd.DataFrame(data_stded, index=data.index,columns=data.columns)
            df_data_stded.reset_index(inplace=True)
        else:
            df_data_stded=data
        return df_data_stded

    def lbl_encode(self, profile_data):
        profile_data.set_index('FACEBOOK_ID', inplace=True)
        for f in profile_data.columns: 
            if f=='AGE_RANGE':
                profile_data[f]=profile_data[f].map({'<= 21':0,'22':1,'23-27':2,'28-30':3,'31-60':4,'>= 61':5})
            else:
                if profile_data[f].dtype=='O': 
                    lbl = preprocessing.LabelEncoder() 
                    lbl.fit(list(profile_data[f].values)) 
                    profile_data[f] = lbl.fit_transform(list(profile_data[f].values))
                else:
                    profile_data[f]=profile_data[f]
        profile_data=profile_data.reset_index()
        return profile_data
