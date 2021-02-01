"""
some useful tools
edit by hichens
"""

from scipy import signal
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings; warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier


## 段波功率
def bandpowers(segment):
    features = []
    for i in range(len(segment)):
        f,Psd = signal.welch(segment[i,:], 100)
        power1 = 0
        power2 = 0
        f1 = []
        for j in range(0,len(f)):
            if(f[j]>=4 and f[j]<=13):
                power1 += Psd[j]
            if(f[j]>=14 and f[j]<=30):
                power2 += Psd[j]
        features.append(power1)
        features.append(power2)
    return features


## 离散余弦变换
from scipy.fftpack import fft, dct
def dct_features(segment):
    features = []
    for i in range(len(segment)):
        dct_coef = dct(segment[i,:], 2, norm='ortho')
        power = sum( j*j for j in dct_coef)
        features.append(power)
    return features


##小波特征
def wavelet_features(epoch):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    features = []
    for i in range(len(epoch)):
        cA,cD=pywt.dwt(epoch[i,:],'coif1')
        cA_values.append(cA)
        cD_values.append(cD)		#calculating the coefficients of wavelet transform.
    for x in range(len(epoch)):   
        cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
        features.append(abs(np.sum(np.square(cA_values[x]))))
        
    for x in range(len(epoch)):      
        cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
        features.append(abs(np.sum(np.square(cD_values[x]))))
        
    return features


def csp_features():
    pass


## test in different models
def test_model(df):
    """
    df: DataFrame format data
    """
    features, labels = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(features, labels, \
        shuffle=True, test_size=0.3, random_state=42)

    ## classify model
    clfs = [
        # Bayes Method
        GaussianNB(priors=None, var_smoothing=1e-9),
        MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None),
        BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None),
        KNeighborsClassifier(n_neighbors=1, 
                            weights='uniform',  # uniform、distance
                            algorithm='auto',  # {‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}
                            leaf_size=10, 
                            # p=1, 
                            metric='minkowski', 
                            metric_params=None, 
                            n_jobs=None),
        
       # you work is add more model
       # ...
    ]
    
    ## train and print(accuracy)
    print("="*40, "result", "="*40)
    print("%40s %10s %10s"%("Model        ", "Accuracy", "time"))
    result = []
    weights = [] # model vote weight
    score_list = []
    train_score_list = []
    for clf in clfs:
        start = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_pred, y_test)
        train_score = clf.score(X_train, y_train)
        score_list.append(score)
        train_score_list.append(train_score)
        weights.append(score)
        print("%40s %5.4f|%5.4f %10.2f s"%(type(clf).__name__, train_score, score, time.time() - start))
        result.append([type(clf).__name__, score])
    print("%40s %5.4f|%5.4f %10.2f s"%("Average", np.mean(train_score_list), np.mean(score_list), 0.0))
    result.append(["Average", np.mean(score)])

    ## model merge
    N = len(weights)
    split_index = sum(weights) / 2
    pred = np.zeros(len(y_pred))
    start = time.time()
    for i, clf in enumerate(clfs):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pred += weights[i] * y_pred
    
    pred[pred<=split_index] = 0
    pred[pred>split_index] = 1
    score = accuracy_score(pred, y_test)
    print("%40s %10.4f %10.2f s"%("Model Stack", score, time.time() - start))
    result.append(["Model Stack", score])
    
    print("="*40, "end", "="*40)
    return result

    
def save_csv(result, path=None):
    res_df = pd.DataFrame(result)
    res_df.columns = ['method', 'score']
    res_df.to_csv(path, index=None)

