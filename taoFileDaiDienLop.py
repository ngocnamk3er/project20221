from operator import index
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from unicodedata import name
from matplotlib.pyplot import axis
import pandas as pd
import io
import numpy as np

# import file dữ liệu

Tk().withdraw()  
train_kahraman = askopenfilename()
print(train_kahraman)
dfData = pd.read_csv(train_kahraman)
print(dfData)
listColumns = dfData.columns.to_list()

print(listColumns)

dfDataTb = pd.DataFrame()

averageSeriesAttr = []

for j in range(len(listColumns)):
    averageSeriesAttr.append([])

columnsLabelName = listColumns[len(listColumns)-1]

print(dfData[columnsLabelName].unique().tolist())

listLabel = dfData[columnsLabelName].unique().tolist()
for i in range(len(listLabel)):
    for j in range(len(listColumns)):
        label =  listLabel[i]
        if j == len(listColumns) - 1:
            averageSeriesAttr[j].append(label)
        else:
            averageSeriesAttr[j].append(round(dfData.loc[dfData[columnsLabelName].eq(label),listColumns[j]].mean(),3))
for j in range(len(listColumns)):
    seri = pd.Series(averageSeriesAttr[j])
    dfDataTb = pd.concat([dfDataTb, seri], axis=1)
dfDataTb.columns = listColumns
print(dfDataTb)
dfDataTb.to_csv('DataTblop.csv',index=False)  
        

# listSTG = list()
# listSCG = list() 
# listSTR = list()
# listLPR = list()
# listPEG = list()
# listUNS = list()
# # Very low
# # print("Very Low------------------------------")
# listUNS.append("Very Low")
# listSTG.append(round(dfData.loc[dfData['UNS'].eq('Very Low'),'STG'].mean(),3))
# listSCG.append(round(dfData.loc[dfData['UNS'].eq('Very Low'),'SCG'].mean(),3))
# listSTR.append(round(dfData.loc[dfData['UNS'].eq('Very Low'),'STR'].mean(),3))
# listLPR.append(round(dfData.loc[dfData['UNS'].eq('Very Low'),'LPR'].mean(),3))
# listPEG.append(round(dfData.loc[dfData['UNS'].eq('Very Low'),'PEG'].mean(),3))



# # print("Low------------------------------")
# listUNS.append("Low")
# listSTG.append(round(dfData.loc[dfData['UNS'].eq('Low'),'STG'].mean(),3))
# listSCG.append(round(dfData.loc[dfData['UNS'].eq('Low'),'SCG'].mean(),3))
# listSTR.append(round(dfData.loc[dfData['UNS'].eq('Low'),'STR'].mean(),3))
# listLPR.append(round(dfData.loc[dfData['UNS'].eq('Low'),'LPR'].mean(),3))
# listPEG.append(round(dfData.loc[dfData['UNS'].eq('Low'),'PEG'].mean(),3))

# # print("Middle------------------------------")
# listUNS.append("Middle")
# listSTG.append(round(dfData.loc[dfData['UNS'].eq('Middle'),'STG'].mean(),3))
# listSCG.append(round(dfData.loc[dfData['UNS'].eq('Middle'),'SCG'].mean(),3))
# listSTR.append(round(dfData.loc[dfData['UNS'].eq('Middle'),'STR'].mean(),3))
# listLPR.append(round(dfData.loc[dfData['UNS'].eq('Middle'),'LPR'].mean(),3))
# listPEG.append(round(dfData.loc[dfData['UNS'].eq('Middle'),'PEG'].mean(),3))

# # print("High------------------------------")
# listUNS.append("High")
# listSTG.append(round(dfData.loc[dfData['UNS'].eq('High'),'STG'].mean(),3))
# listSCG.append(round(dfData.loc[dfData['UNS'].eq('High'),'SCG'].mean(),3))
# listSTR.append(round(dfData.loc[dfData['UNS'].eq('High'),'STR'].mean(),3))
# listLPR.append(round(dfData.loc[dfData['UNS'].eq('High'),'LPR'].mean(),3))
# listPEG.append(round(dfData.loc[dfData['UNS'].eq('High'),'PEG'].mean(),3))

# seriSTG = pd.Series(listSTG)
# seriSCG = pd.Series(listSCG)
# seriSTR = pd.Series(listSTR)
# seriLPR = pd.Series(listLPR)
# seriPEG = pd.Series(listPEG)
# seriUNS = pd.Series(listUNS)

# dfDataTb = pd.concat([dfDataTb, seriSTG], axis=1)
# dfDataTb = pd.concat([dfDataTb, seriSCG], axis=1)
# dfDataTb = pd.concat([dfDataTb, seriSTR], axis=1)
# dfDataTb = pd.concat([dfDataTb, seriLPR], axis=1)
# dfDataTb = pd.concat([dfDataTb, seriPEG], axis=1)
# dfDataTb = pd.concat([dfDataTb, seriUNS], axis=1)

# print(dfDataTb)


