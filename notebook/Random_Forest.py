# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:51:57 2019

@author: Vladimir
"""


import argparse
import numpy as np
import pandas as pd
import sklearn as sk


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

#df=pd.read_csv(args.infile)

df=pd.read_csv('C:/TSU_GIT/MedicalDataAnalysisService/R_scripts/Arizona_informative.csv')


label_feature = df.Class

df=df.drop(['Class'], axis=1)



X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(df, label_feature, 
                 test_size=args.cross, stratify= label_feature)

           

model= sk.ensemble.RandomForestClassifier()  (iterations=args.itera, depth=6, learning_rate=args.learn, 
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
