# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import argparse
import catboost
from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


parser = argparse.ArgumentParser(description='Process some integers.')
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', help='Enter initional file')
parser.add_argument('-o', '--outfile', help='Enter path to outfile')
parser.add_argument('-oo', '--ooutfile', help='Enter path to outfile second')
parser.add_argument('-it', '--itera', type=int, help='Count of iterations', default=10)
parser.add_argument('-l', '--learn', type=float, help='Count of learning rate', default=0.1)
parser.add_argument('-c', '--cross', type=float, help='Count of test train split', default=0.5)
args = parser.parse_args()



#'C:/TSU_GIT/MedicalDataAnalysisService/R_scripts/Arizona_informative.csv'

df=pd.read_csv(args.infile)

#df=pd.read_csv('C:/TSU_GIT/MedicalDataAnalysisService/R_scripts/Arizona_informative.csv')


# кодирование признака в цифры 
le = LabelEncoder()
encode_feature = le.fit_transform(df.Class)



df=df.drop(['Class'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(df, encode_feature, 
                 test_size=args.cross, stratify= encode_feature)

train_pool= Pool(data=X_train, label= y_train)

test_pool= Pool(data=X_test, label= y_test)             

model=CatBoostClassifier(iterations=args.itera, depth=6, learning_rate=args.learn, 
                         loss_function='MultiClass', eval_metric= 'Accuracy', logging_level='verbose')



model.fit(train_pool, eval_set=test_pool, logging_level='Verbose')

pred= model.predict(data=test_pool, prediction_type='Class')

acc_test=accuracy_score(y_test, pred)

kappa_test=cohen_kappa_score(y_test, pred)


pred= model.predict(data=train_pool, prediction_type='Class')

acc_train=accuracy_score(y_train, pred)

kappa_train=accuracy_score(y_train, pred)



d = pd.DataFrame(data={'Accuracy': [acc_train, acc_test], 'Kappa': [kappa_train, kappa_test]})

df1=d.rename(index={0:'train', 1:'test'})


#prob= model.predict_proba(data=test_pool)

pp=model.get_evals_result()

dl = pd.DataFrame(data=pp['learn'])

dv = pd.DataFrame(data=pp['validation_0'])

result = pd.concat([dl, dv], axis=1, sort=False)

result.columns = ['Acc_learn', 'Multi_learn', 'Acc_val', 'Multi_val']


result.to_csv(path_or_buf=args.outfile, index=False)

df1.to_csv(path_or_buf=args.ooutfile, index=False)



                 
                 
                 
                 
                 