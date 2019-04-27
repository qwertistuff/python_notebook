# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""


import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



import Block_utilities as f

print (f.joke())
 
print (f.sum())


f.graph_score()

df1=pd.read_csv('C:/TSU_GIT/MedicalDataAnalysisService/R_scripts/fgdfg.csv')

result=f.transform_result_to_graph(df1)

f.graph_metrics(result)



