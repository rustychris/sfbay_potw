"""
Basic data quality check on the compiled data - 
 1. does each station have the data we expect?
 2. do the units appear to be correct?

"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

## 
csvs=glob.glob('compiled_inputs/*.csv')
csvs=[s for s in csvs if 'stormwater' not in s]
names=[os.path.basename(fn).replace('.csv','')
       for fn in csvs]
dfs=[]
for fn in csvs:
    dfs.append( pd.read_csv(fn,parse_dates=['Date']) )


## 

parms=['flow mgd',
       'NO3 mg/L N',
       'NH3 mg/L N',
       'PO4 mg/L P']

for parmi,parm in enumerate(parms):
    plt.figure(parmi).clf()
    fig,ax=plt.subplots(1,1,num=parmi)

    for dfi,df in enumerate(dfs):
        ax.plot_date(df.Date,df[parm],ls='-',label=names[dfi])
    ax.set_ylabel(parm)

# nothing really bizarre.

## 

plt.close('all')

## 

# one figure per source

parms=['flow mgd',
       'NO3 mg/L N',
       'NH3 mg/L N',
       'PO4 mg/L P']

for dfi,df in enumerate(dfs):
    plt.figure(dfi).clf()
    fig,axs=plt.subplots(len(parms),1,num=dfi,sharex=True)
    axs[0].set_title(names[dfi])

    for parmi,parm in enumerate(parms):
        axs[parmi].plot_date(df.Date,df[parm],ls='-')
        axs[parmi].set_ylabel(parm)

## 
