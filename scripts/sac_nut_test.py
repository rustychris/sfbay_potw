"""
Clean source data from wastewater treatment plants, writing 
intermediate csv files with uniform naming conventions, units,
date formats, etc.

Limited to Delta plants.
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr

from stompy import utils # for date utilities

### Utility functions

def load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname):
    dat = pd.read_csv(filename, usecols=cols)
    time = [dt.datetime.strptime(dat[tname][i], tformat) \
            for i in range(len(dat[tname])) \
            if (dat[unitsname][i] == units and dat[method][i] == analyte)]
    var = [dat[varname][i] for i in range(len(dat[varname])) \
           if (dat[unitsname][i] == units and dat[method][i] == analyte)]

    var = np.asarray(var)
    d = {'Time': pd.Series(time),
         vname: pd.Series(var)}
    df = pd.DataFrame(d)
    return df

def day_ind(time):
    d = 0 
    ind = np.zeros(len(time))
    for i in range(1,len(time)):
        if time[i].year == time[i-1].year and time[i].month == time[i-1].month and time[i].day == time[i-1].day:
            ind[i] = d
        else:
            d += 1
            ind[i] = d
    return ind

def day_avg(time, var, ind):
    date = []
    dvar = np.zeros(int(ind[-1]))
    d = 0 
    for i in range(int(ind[-1])):
        date.append(time[d].to_pydatetime().date())
        dvar[i] = np.mean(var[np.where(ind==i)[0]])
        d += len(np.where(ind==i)[0])
    return date, dvar
    
def var_fields(date, variable):
    # This used to loop through, but since all datetimes should be datetime64 now,
    # just do it in one go:
    valid=~variable.isnull()
    return date[valid],variable[valid]

def data_fill(colname1, colname2, df):
    col1 = df[colname1]
    col2 = df[colname2]
    ind = np.where(pd.isnull(col1))
    col1[ind[0]] = col2[ind[0]]
    return df

### variables that don't change between wwtp files

inpath = "../sources/delta_sources/"    
outpath = "../outputs/intermediate/delta/"
tformat = "%m/%d/%Y %H:%M"
varname = "Result"
unitsname = "Units"




filename = inpath + "A0210451.csv"
outfile = outpath + "Sac_NO3.csv"
cols = [4,5,7,9]
tname = "Collection Date"
method = "Analyte"
analyte = "Dissolved Nitrate"
units = "mg/L"
vname = "NO3 mg/L N"
dat = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname)
ind = day_ind(am_df.Time)
valid = np.where(dat["NO3 mg/L N"] != "< R.L.")
tvalid = dat["Time"][valid[0]]
dvalid = dat["NO3 mg/L N"][valid[0]]
ivalid = ind[valid[0]]

date = []
dvar = np.zeros(int(ivalid[-1]))
d = 0 
for i in range(int(ivalid[-1])):
    date.append(tvalid[d].to_pydatetime().date())
    dvar[i] = np.mean(dvalid[np.where(ivalid==i)[0]])
    d += len(np.where(ivalid==i)[0])




time, no3 = day_avg(tvalid, dvalid, ind)
d = {'Date': pd.Series(time),
     vname: pd.Series(no3)}
df1 = pd.DataFrame(d)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time, am, '-x', color="darkslategray")
ax.set_title("Sacramento River : Dissolved Nitrate (N)")
ax.set_ylabel("mg/L")    