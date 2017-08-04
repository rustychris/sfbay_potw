"""
Clean source data for Sacramento River. 
Nutrient data from DWR data and Flow from USGS data.
Must run this script after usgs_data_loading.py to get usgs flow data from intermediate csv. 
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr

from stompy import utils # for date utilities

### Utility functions

# Read once to speed things up
inpath = "../sources/delta_sources/"    
cols = range(17) # include more, so we get reporting limit, too.
tname = "Collection Date"
filename = inpath + "A0210451.csv"
parsed_dwr=pd.read_csv(filename, usecols=cols, parse_dates=[tname])

def load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname):
    # copy the pre-parsed dataframe, so any modifications here don't contaminate the
    # original data.
    dat = parsed_dwr.copy()

    # Replace low values with half the reporting limit
    low = (dat.Result=='< R.L.')
    dat.loc[low, 'Result'] = 0.5 * dat.loc[low, 'Rpt Limit'].astype(np.float64)

    # Narrow down to the desired analyte, and just the time and results columns
    dat2 = dat.loc[ dat[method]==analyte, [tname,'Result'] ]
    dat3 = dat2.rename(columns={'Result':vname,
                                tname:'Time'})
    # should have the below-reporting-limit values taken care of, and can
    # convert to floats
    dat3[vname]=dat3[vname].astype(np.float64)
    return dat3
    # time = [dt.datetime.strptime(dat[tname][i], tformat) \
    #         for i in range(len(dat[tname])) \
    #         if (dat[unitsname][i] == units and dat[method][i] == analyte)]
    # var = [dat[varname][i] for i in range(len(dat[varname])) \
    #        if (dat[unitsname][i] == units and dat[method][i] == analyte)]
    # 
    # var = np.asarray(var)
    # d = {'Time': pd.Series(time),
    #      vname: pd.Series(var)}
    # df = pd.DataFrame(d)
    # return df

def day_ind(time):
    """ Given a pandas Series of datetimes, return an integer array with
    indices grouping consecutive datetimes on the same day.
    """
    d = 0 
    ind = np.zeros(len(time))
    for i in range(1,len(time)):
        if ( (time.iloc[i].year == time.iloc[i-1].year) and
             (time.iloc[i].month == time.iloc[i-1].month) and
             (time.iloc[i].day == time.iloc[i-1].day)):
            ind[i] = d
        else:
            d += 1
            ind[i] = d
    return ind


### variables that don't change between analytes

outpath = "../outputs/intermediate/delta/"
tformat = "%m/%d/%Y %H:%M"
varname = "Result"
unitsname = "Units"

method = "Analyte"
units = "mg/L"
analyte = "Dissolved Nitrate"
vname = "NO3 mg/L N"
dat = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname)

# The nitrate data is actually in NO3 mg/l, not as N.  Convert here:
dat[vname] *= 14/62.

# When multiple samples occur on the same day, average:
dat2=dat.groupby(dat.Time.dt.date).mean().reset_index()
df1=dat2.rename(columns={'Time':'Date'})

# Older code which removed below-reporting-limit values, and averaged
# within days.

# ind = day_ind(dat.Time)
# valid = np.where(dat["NO3 mg/L N"] != "< R.L.")
# tvalid = dat["Time"][valid[0]].reset_index()
# del tvalid["index"]
# dvalid = dat["NO3 mg/L N"][valid[0]].reset_index()
# del dvalid["index"]
# ivalid = ind[valid[0]]
# 
# date = []
# dvar = np.zeros(int(ivalid[-1]))
# d = 0 
# for i in range(int(ivalid[-1])):
#     date.append(tvalid["Time"][d].to_pydatetime().date())
#     dvar[i] = np.mean(dvalid["NO3 mg/L N"][np.where(ivalid==i)[0]].astype(float))
#     d += len(np.where(ivalid==i)[0])
# time = date
# no3 = dvar
# d = {'Date': pd.Series(time),
#      vname: pd.Series(no3)}
# df1 = pd.DataFrame(d)


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(df1.Date, df1[vname], '-x', color="darkslategray")
ax.set_title("Sacramento River : Dissolved Nitrate (N)")
ax.set_ylabel("mg/L N")


units = "mg/L as N"
analyte = "Dissolved Nitrate + Nitrite"
vname = "N+N mg/L N"
dat = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname)
# When multiple samples occur on the same day, average:
dat2=dat.groupby(dat.Time.dt.date).mean().reset_index()
df2=dat2.rename(columns={'Time':'Date'})


# ind = day_ind(dat.Time)
# valid = np.where(dat["N+N mg/L N"] != "< R.L.")
# tvalid = dat["Time"][valid[0]].reset_index()
# del tvalid["index"]
# dvalid = dat["N+N mg/L N"][valid[0]].reset_index()
# del dvalid["index"]
# ivalid = ind[valid[0]]
# 
# date = []
# dvar = np.zeros(int(ivalid[-1]))
# d = 0 
# for i in range(int(ivalid[-1])):
#     date.append(tvalid["Time"][d].to_pydatetime().date())
#     dvar[i] = np.mean(dvalid["N+N mg/L N"][np.where(ivalid==i)[0]].astype(float))
#     d += len(np.where(ivalid==i)[0])
# time = date
# nn = dvar
# d = {'Date': pd.Series(time),
#      vname: pd.Series(nn)}
# df2 = pd.DataFrame(d)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(df2.Date, df2[vname], '-x', color="darkslategray")
ax.set_title("Sacramento River : Dissolved Nitrate + Nitrite (N)")
ax.set_ylabel("mg/L N")

sac = pd.merge(df1, df2, how='outer', on='Date')


units = "mg/L as P"
analyte = "Dissolved Ortho-phosphate"
vname = "o-PO4 mg/L P"
dat = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname)
# When multiple samples occur on the same day, average:
dat2=dat.groupby(dat.Time.dt.date).mean().reset_index()
df3=dat2.rename(columns={'Time':'Date'})

# ind = day_ind(dat.Time)
# valid = np.where(dat["o-PO4 mg/L P"] != "< R.L.")
# tvalid = dat["Time"][valid[0]].reset_index()
# del tvalid["index"]
# dvalid = dat["o-PO4 mg/L P"][valid[0]].reset_index()
# del dvalid["index"]
# ivalid = ind[valid[0]]
# 
# date = []
# dvar = np.zeros(int(ivalid[-1]))
# d = 0 
# for i in range(int(ivalid[-1])):
#     date.append(tvalid["Time"][d].to_pydatetime().date())
#     dvar[i] = np.mean(dvalid["o-PO4 mg/L P"][np.where(ivalid==i)[0]].astype(float))
#     d += len(np.where(ivalid==i)[0])
# time = date
# po4 = dvar
# d = {'Date': pd.Series(time),
#      vname: pd.Series(po4)}
# df3 = pd.DataFrame(d)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(df3.Date, df3[vname], '-x', color="darkslategray")
ax.set_title("Sacramento River : Dissolved Ortho-phosphate (P)")
ax.set_ylabel("mg/L P")


sac = pd.merge(sac, df3, how='outer', on='Date')


units = "mg/L as P"
analyte = "Total Phosphorus"
vname = "TP mg/L P"
dat = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname)
dat2=dat.groupby(dat.Time.dt.date).mean().reset_index()
df4=dat2.rename(columns={'Time':'Date'})


# ind = day_ind(dat.Time)
# valid = np.where(dat["TP mg/L P"] != "< R.L.")
# tvalid = dat["Time"][valid[0]].reset_index()
# del tvalid["index"]
# dvalid = dat["TP mg/L P"][valid[0]].reset_index()
# del dvalid["index"]
# ivalid = ind[valid[0]]
# 
# date = []
# dvar = np.zeros(int(ivalid[-1]))
# d = 0 
# for i in range(int(ivalid[-1])):
#     date.append(tvalid["Time"][d].to_pydatetime().date())
#     dvar[i] = np.mean(dvalid["TP mg/L P"][np.where(ivalid==i)[0]].astype(float))
#     d += len(np.where(ivalid==i)[0])
# time = date
# tp = dvar
# d = {'Date': pd.Series(time),
#      vname: pd.Series(tp)}
# df4 = pd.DataFrame(d)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(df4.Date, df4[vname], '-x', color="darkslategray")
ax.set_title("Sacramento River : Total Phosphorus (P)")
ax.set_ylabel("mg/L P")


sac = pd.merge(sac, df4, how='outer', on='Date')


units = "mg/L as N"
analyte = "Dissolved Ammonia"
vname = "NH3 mg/L N"
dat = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, analyte, cols, vname)

dat2=dat.groupby(dat.Time.dt.date).mean().reset_index()
df5=dat2.rename(columns={'Time':'Date'})

# ind = day_ind(dat.Time)
# valid = np.where(dat["NH3 mg/L N"] != "< R.L.")
# tvalid = dat["Time"][valid[0]].reset_index()
# del tvalid["index"]
# dvalid = dat["NH3 mg/L N"][valid[0]].reset_index()
# del dvalid["index"]
# ivalid = ind[valid[0]]
# 
# date = []
# dvar = np.zeros(int(ivalid[-1]))
# d = 0 
# for i in range(int(ivalid[-1])):
#     date.append(tvalid["Time"][d].to_pydatetime().date())
#     dvar[i] = np.mean(dvalid["NH3 mg/L N"][np.where(ivalid==i)[0]].astype(float))
#     d += len(np.where(ivalid==i)[0])
# time = date
# nh3 = dvar
# d = {'Date': pd.Series(time),
#      vname: pd.Series(nh3)}
# df5 = pd.DataFrame(d)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(df5.Date, df5[vname], '-x', color="darkslategray")
ax.set_title("Sacramento River : Dissolved Ammonia (N)")
ax.set_ylabel("mg/L N")


sac = pd.merge(sac, df5, how='outer', on='Date')

# merge in flow from verona (in csv with nutrient data from freeport)
dat = pd.read_csv("../outputs/intermediate/delta/sacramento_at_freeport.csv")
d = {'Date': pd.Series(dat["Date"]),
     'flow mgd': pd.Series(dat["flow mgd"])}
df6 = pd.DataFrame(d)
sac = pd.merge(sac, df6, how='outer', on='Date')

sac['Date']=utils.to_dt64(sac.Date.values) # standardize type for Date
# save final concatenated file
sac.to_csv(outpath+"sacramento_at_verona.csv")

plt.close('all') # helps with memory use

