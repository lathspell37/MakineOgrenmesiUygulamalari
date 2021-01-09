import seaborn as sns
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split





def okumacsv(name):
    oku = pd.read_csv(name)
    df = oku.copy()
    df = df.dropna()
    return df

def aim(df,a):
    y = df[aim]
    X = df.drop([a],axis=1)
    return X,y

def encode(X):
    return pd.get_dummies(X)

def aykiriBaskilama(df):
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    clf.fit_predict(df)
    df_scores = clf.negative_outlier_factor_
    esik = np.sort(df_scores)[13]
    aykiri = df_scores > esik
    yeni = df[df_scores > esik]
    baski = df[df_scores == esik]
    aykirilar = df[~aykiri]

    res = aykirilar.to_records(index=False)
    res[:] = baski.to_records(index=False)
    df[~aykiri] = pd.DataFrame(res, index=df[~aykiri].index)
    print(df[~aykiri])
    return df

def split(x,y,a,b):
    return train_test_split(x, y, test_size= a, random_state=b)

def normalizasyon(df):
    return preprocessing.normalize(df)

def standartizasyon(df):
    return preprocessing.scale(df)

def minmax(df,x,y):
    scaler = preprocessing.MinMaxScaler(feature_range=(x, y))
    return scaler.fit_transform(df)





veriSeti = okumacsv('SkillCraft1_Dataset.csv')
veriSeti = pd.get_dummies(veriSeti)
print(minmax(veriSeti,12,22))












