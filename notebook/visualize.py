# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

import argparse

def makeHtmlFromImage(imagePath, outputFolderPath):
    import base64
    image = open(imagePath, "rb")
    imageBytes = image.read()
    imageBase64 = base64.b64encode(imageBytes).decode()
    htmlString = '<img src="data:image/png;base64, {}"></img>'.format(imageBase64)
    htmlFile = open(outputFolderPath + "mainResult.html", "w")
    htmlFile.write(htmlString)
    htmlFile.close()


parser = argparse.ArgumentParser(description='Process some integers.')
parser = argparse.ArgumentParser()

parser.add_argument('-in', '--infile', help='Enter initional file')
parser.add_argument('-out', '--outfile', help='Enter path to outfile')


args = parser.parse_args()

df = pd.read_csv(args.infile)

#df = pd.read_csv('C:/TSU_GIT/MedicalDataAnalysisService/R_scripts/tsne_sjat.csv')


gg=sns.relplot(x='V1', y='V2', hue='Class', data=df)

#gg=sns.relplot(x=df.iloc[:,1], y=df.iloc[:,2], hue=df.iloc[:,0], data=df)

gg.savefig( args.outfile + 'result_tsne.png', dpi=300)

#gg.savefig( 'C:/TSU_GIT/MedicalDataAnalysisService/R_scripts/' + 'result_tsne.png', dpi=300)

plt.clf()

makeHtmlFromImage(args.outfile + 'result_tsne.png',args.outfile)