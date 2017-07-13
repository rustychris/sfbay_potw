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

def load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname):
    dat = pd.read_csv(filename, usecols=cols)
    time = [dt.datetime.strptime(dat[tname][i], tformat) \
            for i in range(len(dat[tname])) \
            if (dat[unitsname][i] == units and pd.isnull(dat[method][i]))]
    var = [dat[varname][i] for i in range(len(dat[varname])) \
           if (dat[unitsname][i] == units and pd.isnull(dat[method][i]))]

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
tname = "Sampling Date Time"
tformat = "%m/%d/%Y %H:%M"
varname = "Result"
unitsname = "Units"
method = "Calculated Method"
cols = [3,5,6,12]
prat = 2.23

### DAVIS

# ammonia
filename = inpath + "Davis_5A570100001_Ammonia_Total_as_N.csv"
outfile = outpath + "Davis_Ammonia.csv"
units = "mg/L"
vname = "NH3 mg/L N"
am_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(am_df.Time)
time, am = day_avg(am_df.Time, am_df[vname], ind)
d = {'Date': pd.Series(time),
     vname: pd.Series(am)}
df1 = pd.DataFrame(d)
#df1.to_csv(outfile)
# plotting
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time, am, '-x', color="darkslategray")
ax.set_title("Davis: Total Ammonia (as N)")
ax.set_ylabel("mg/L")    
fig.savefig(outpath + "figures/Davis_Ammonia.png") 

# flow
filename = inpath + "Davis_5A570100001_Flow.csv"
outfile = outpath + "Davis_Flow.csv"
units = "MGD"
vname = "flow mgd"
flw_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(flw_df.Time)
time, flw = day_avg(flw_df.Time, flw_df[vname], ind)
d = {'Date': pd.Series(time),
     vname: pd.Series(flw)}
df2 = pd.DataFrame(d)
#df2.to_csv(outfile)
# plotting
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time, flw, '-x', color="cadetblue")
ax.set_title("Davis: Flow")
ax.set_ylabel("MGD")
fig.savefig(outpath + "figures/Davis_Flow.png")

# begin concatenate 
davis = pd.merge(df1, df2, how='outer', on='Date')

# nitrate
filename = inpath + "Davis_5A570100001_Nitrate_Total_as_N.csv"
outfile = outpath + "Davis_Nitrate.csv"
units = "mg/L"
vname = "NO3 mg/L N"
na_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(na_df.Time)
time, na = day_avg(na_df.Time, na_df[vname], ind)
d = {'Date': pd.Series(time),
     vname: pd.Series(na)}
df3 = pd.DataFrame(d)
#df3.to_csv(outfile)
# plotting
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time, na, '-x', color="seagreen")
ax.set_title("Davis: Total Nitrate (as N)")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Davis_Nitrate.png")    

# concatenate
davis = pd.merge(davis, df3, how='outer', on='Date')

# nitrite
filename = inpath + "Davis_5A570100001_Nitrite_Total_as_N.csv"
outfile = outpath + "Davis_Nitrite.csv"
units = "mg/L"
vname = "NO2 mg/L N"
ni_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(ni_df.Time)
time, ni = day_avg(ni_df.Time, ni_df[vname], ind)
d = {'Date': pd.Series(time),
     vname: pd.Series(ni)}
df4 = pd.DataFrame(d)
#df4.to_csv(outfile)
# plotting
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time, ni, '-x', color="royalblue")
ax.set_title("Davis: Total Nitrite (as N)")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Davis_Nitrite.png")

# concatenate
davis = pd.merge(davis, df4, how='outer', on='Date')

# nitrite + nitrate
filename = inpath + "Davis_5A570100001_Nitrite+Nitrate_as_N.csv"
outfile = outpath + "Davis_Nitrite+Nitrate.csv"
units = "mg/L"
vname = "N+N mg/L N"
nn_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(nn_df.Time)
time, nn = day_avg(nn_df.Time, nn_df[vname], ind)
d = {'Date': pd.Series(time),
     vname: pd.Series(nn)}
df5 = pd.DataFrame(d)
#df5.to_csv(outfile)
# plotting
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time, nn, '-x', color="olivedrab")
ax.set_title("Davis: Nitrite+Nitrate (as N)")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Davis_Nitrite+Nitrate.png")    

# concatenate
davis = pd.merge(davis, df5, how='outer', on='Date')

# phosphorus
filename = inpath + "Davis_5A570100001_Phosphorus_Total_as_P.csv"
outfile = outpath + "Davis_Phosphorus.csv"
units = "mg/L"
vname = "P mg/L P"
p_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(p_df.Time)
time, p = day_avg(p_df.Time, p_df[vname], ind)
d = {'Date': pd.Series(time),
     vname: pd.Series(p),
     'PO4 mg/L P': pd.Series(p/prat)}
df6 = pd.DataFrame(d)
#df6.to_csv(outfile)
# plotting
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time, p, '-x', color="indianred")
ax.set_title("Davis: Total Phosphorus (as P)")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Davis_Phosphorus.png")    

# concatenate
davis = pd.merge(davis, df6, how='outer', on='Date')
davis['Date']=utils.to_dt64(davis.Date.values) # standardize type for Date
# save final concatenated file
davis.to_csv(outpath+"davis.csv")

plt.close('all') # helps with memory use

### MANTECA1 - CWIQS data
if 0: # Manteca1.csv is no longer generated.  Using Manteca.csv at bottom of this script instead.
    # ammonia
    filename = inpath + "Manteca_5B390104001_Ammonia_Total_as_N.csv"
    outfile = outpath + "Manteca_Ammonia.csv"
    units = "mg/L"
    vname = "NH3 mg/L N"
    am_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(am_df.Time)
    time, am = day_avg(am_df.Time, am_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(am)}
    df1 = pd.DataFrame(d)
    #df1.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, am, '-x', color="darkslategray")
    ax.set_title("Manteca: Total Ammonia (as N)")
    ax.set_ylabel("mg/L")    
    fig.savefig(outpath + "figures/Manteca_Ammonia.png")
    
    # flow
    filename = inpath + "Manteca_5B390104001_Flow.csv"
    outfile = outpath + "Manteca_Flow.csv"
    units = "MGD"
    vname = "flow mgd"
    flw_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(flw_df.Time)
    time, flw = day_avg(flw_df.Time, flw_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(flw)}
    df2 = pd.DataFrame(d)
    #df2.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, flw, '-x', color="cadetblue")
    ax.set_title("Manteca: Flow")
    ax.set_ylabel("MGD")
    fig.savefig(outpath + "figures/Manteca_Flow.png")
    
    # begin concatenate
    manteca = pd.merge(df1, df2, how='outer', on='Date')
    
    # nitrate
    filename = inpath + "Manteca_5B390104001_Nitrate_Total_as_N.csv"
    outfile = outpath + "Manteca_Nitrate.csv"
    units = "mg/L"
    vname = "NO3 mg/L N"
    na_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(na_df.Time)
    time, na = day_avg(na_df.Time, na_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(na)}
    df3 = pd.DataFrame(d)
    #df3.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, na, '-x', color="seagreen")
    ax.set_title("Manteca: Total Nitrate (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Manteca_Nitrate.png")    
    
    # concatenate
    manteca = pd.merge(manteca, df3, how='outer', on='Date')
    
    # nitrite
    filename = inpath + "Manteca_5B390104001_Nitrite_Total_as_N.csv"
    outfile = outpath + "Manteca_Nitrite.csv"
    units = "mg/L"
    vname = "NO2 mg/L N"
    ni_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(ni_df.Time)
    time, ni = day_avg(ni_df.Time, ni_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(ni)}
    df4 = pd.DataFrame(d)
    #df4.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, ni, '-x', color="royalblue")
    ax.set_title("Manteca: Total Nitrite (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Manteca_Nitrite.png")
    
    # concatenate
    manteca = pd.merge(manteca, df4, how='outer', on='Date')
    
    # nitrite + nitrate
    filename = inpath + "Manteca_5B390104001_Nitrite+Nitrate_as_N.csv"
    outfile = outpath + "Manteca_Nitrite+Nitrate.csv"
    units = "mg/L"
    vname = "N+N mg/L N"
    nn_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(nn_df.Time)
    time, nn = day_avg(nn_df.Time, nn_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(nn)}
    df5 = pd.DataFrame(d)
    #df5.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, nn, '-x', color="olivedrab")
    ax.set_title("Manteca: Nitrite+Nitrate (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Manteca_Nitrite+Nitrate.png")    
    
    # concatenate
    manteca = pd.merge(manteca, df5, how='outer', on='Date')
    
    # nitrogen
    filename = inpath + "Manteca_5B390104001_Nitrogen_Total_as_N.csv"
    outfile = outpath + "Manteca_Nitrogen.csv"
    units = "mg/L"
    vname = "N mg/L N"
    n_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(n_df.Time)
    time, n = day_avg(n_df.Time, n_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(n)}
    df6 = pd.DataFrame(d)
    #df6.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, n, '-x', color="slateblue")
    ax.set_title("Manteca: Total Nitrogen (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Manteca_Nitrogen.png")    
    
    # concatenate
    manteca = pd.merge(manteca, df6, how='outer', on='Date')
    
    # save final concatenated file
    manteca.to_csv(outpath+"manteca1.csv")

    plt.close('all')
    
### TRACY

if 1: # Tracy, part A
    # ammonia
    filename = inpath + "Tracy_5B390108001_Ammonia_Total_as_N.csv"
    outfile = outpath + "Tracy_Ammonia.csv"
    units = "mg/L"
    vname = "NH3 mg/L N"
    am_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(am_df.Time)
    time, am = day_avg(am_df.Time, am_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(am)}
    df1 = pd.DataFrame(d)
    #df1.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, am, '-x', color="darkslategray")
    ax.set_title("Tracy: Total Ammonia (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Ammonia.png")    
    
    # flow
    filename = inpath + "Tracy_5B390108001_Flow.csv"
    outfile = outpath + "Tracy_Flow.csv"
    units = "MGD"
    vname = "flow mgd"
    flw_df = load_wwtp(filename=filename, tname=tname, tformat=tformat, varname=varname, unitsname=unitsname, units=units, method="Analytical Method", cols=[2,5,6,12], vname=vname)
    ind = day_ind(flw_df.Time)
    time, flw = day_avg(flw_df.Time, flw_df[vname], ind)
    d = {'Date': pd.Series(time),
             vname: pd.Series(flw)}
    df2 = pd.DataFrame(d)
    #df2.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, flw, '-x', color="cadetblue")
    ax.set_title("Tracy: Flow")
    ax.set_ylabel("MGD")
    fig.savefig(outpath + "figures/Tracy_Flow.png")
    
    # concatenate
    tracy = pd.merge(df1, df2, how='outer', on='Date')
    
    # nitrate
    filename = inpath + "Tracy_5B390108001_Nitrate_Total_as_N.csv"
    outfile = outpath + "Tracy_Nitrate.csv"
    units = "mg/L"
    vname = "NO3 mg/L N"
    na_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(na_df.Time)
    time, na = day_avg(na_df.Time, na_df[vname], ind)
    d = {'Date': pd.Series(time),
             vname: pd.Series(na)}
    df3 = pd.DataFrame(d)
    #df3.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, na, '-x', color="seagreen")
    ax.set_title("Tracy: Total Nitrate (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Nitrate.png")
    
    # concatenate
    tracy = pd.merge(tracy, df3, how='outer', on='Date')
    
    # nitrite
    filename = inpath + "Tracy_5B390108001_Nitrite_Total_as_N.csv"
    outfile = outpath + "Tracy_Nitrite.csv"
    units = "mg/L"
    vname = "NO2 mg/L N"
    ni_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(ni_df.Time)
    time, ni = day_avg(ni_df.Time, ni_df[vname], ind)
    d = {'Date': pd.Series(time),
             vname: pd.Series(ni)}
    df4 = pd.DataFrame(d)
    #df4.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, ni, '-x', color="royalblue")
    ax.set_title("Tracy: Total Nitrite (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Nitrite.png")
    
    # concatenate
    tracy = pd.merge(tracy, df4, how='outer', on='Date')
    
    # nitrite + nitrate
    filename = inpath + "Tracy_5B390108001_Nitrite+Nitrate_as_N.csv"
    outfile = outpath + "Tracy_Nitrite+Nitrate.csv"
    units = "mg/L"
    vname = "N+N mg/L N"
    nn_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(nn_df.Time)
    time, nn = day_avg(nn_df.Time, nn_df[vname], ind)
    d = {'Date': pd.Series(time),
             vname: pd.Series(nn)}
    df5 = pd.DataFrame(d)
    #df5.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, nn, '-x', color="olivedrab")
    ax.set_title("Tracy: Nitrite+Nitrate (as N)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Nitrite+Nitrate.png")        
    
    # concatenate
    tracy = pd.merge(tracy, df5, how='outer', on='Date')
    
    # phosphorus
    filename = inpath + "Tracy_5B390108001_Phosphorus_Total_as_P.csv"
    outfile = outpath + "Tracy_Phosphorus.csv"
    units = "mg/L"
    vname = "TP mg/L P" # had been just P, but seems more consistent to call this TP
    p_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
    ind = day_ind(p_df.Time)
    time, p = day_avg(p_df.Time, p_df[vname], ind)
    d = {'Date': pd.Series(time),
         vname: pd.Series(p),
         'PO4 mg/L P': pd.Series(p/prat)}
    df6 = pd.DataFrame(d)
    #df6.to_csv(outfile)
    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, p, '-x', color="indianred")
    ax.set_title("Tracy: Total Phosphorus (as P)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Phosphorus.png")    
    
    # concatenate - this is just part of the Tracy data, to be merged
    # with a second tracy dataset below
    tracyA = pd.merge(tracy, df6, how='outer', on='Date')

    # TRACY, part B
    filename = inpath + "tracy.csv"
    tracyB = pd.read_csv(filename,parse_dates=['Date'])
    tracyB.rename(columns={'Flow': 'flow mgd', 'NH3': 'NH3 mg/L N', 
                           'NO3': 'NO3 mg/L N', 'TP': 'TP mg/L P'}, inplace=True)
    
    # TRACY, merge A and B
    # at this point, tracyA has dates as python datetime, tracyB has them as numpy
    # datetime.
    
    outfile = outpath + "tracy.csv"

    # make a copy of tracyA to allow safely changing the datatype of Date.
    tracyA2=tracyA.copy()
    tracyA2['Date']=utils.to_dt64(tracyA2['Date'].values)

    tracy=pd.merge(tracyA2,tracyB,how='outer',on='Date')
    tracy.sort_values('Date',inplace=True) # merge does not guarantee sorted output

    # tracyA2 comes in with         Date  NH3 mg/L N  flow mgd  NO3 mg/L N  NO2 mg/L N  N+N mg/L N 
    #                               P mg/L P  PO4 mg/L P
    # It is from CWIQS data

    # tracyB comes in with          Date  flow mgd  NH3 mg/L N  NO3 mg/L N  TP mg/L P
    # This is from 
    # Go analyte by analyte to see what's going on:

    ## 
    y_missing=tracy['NH3 mg/L N_y'].isnull()
    tracy['NH3 mg/L N']=np.where( ~y_missing,
                                  tracy['NH3 mg/L N_y'],
                                  tracy['NH3 mg/L N_x'])

    if 1: # compare NH3
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        ax.plot(tracy.Date,tracy['NH3 mg/L N_x'],marker='x',label='NH3 CWIQS')
        ax.plot(tracy.Date,tracy['NH3 mg/L N_y'],marker='x',label='NH3 Novick')
        ax.plot(tracy.Date,tracy['NH3 mg/L N'],marker='o',mfc='none',label='NH3 out',lw=0.9,c='k')
        ax.legend()
        # y is longer, and has better coverage, with just a few more recent points
        # coming from only x
        # overall concentrations quite similar.
        fig.savefig(outpath+"figures/Tracy-nh3-compare_sources.png")

    ##

    y_missing=tracy['NO3 mg/L N_y'].isnull()
    tracy['NO3 mg/L N']=np.where( ~y_missing,
                                  tracy['NO3 mg/L N_y'],
                                  tracy['NO3 mg/L N_x'])

    if 1: # compare NO3
        # on the occasions that there is data from both (mid-2010 to early 2013)
        # 'x' (A2) is about 1/3 the value of 'y' (B).  The higher values line up
        # with previous levels and later levels, and even later levels from A2,
        # suggesting that we should go with the B values when both are present.
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        ax.plot(tracy.Date,tracy['NO3 mg/L N_x'],marker='x',label='NO3 CWIQS')
        ax.plot(tracy.Date,tracy['NO3 mg/L N_y'],marker='x',label='NO3 Novick')
        ax.plot(tracy.Date,tracy['NO3 mg/L N'],marker='o',mfc='none',label='NO3 out',lw=0.9,c='k')
        ax.legend()
        fig.savefig(os.path.join(outpath,"figures/Tracy-NO3-compare_sources.png"))

    # NO2 only from tracyA2

    # NN only from tracyA2, only 2013--on
    # NO2, where present, is much less than 10% of NO3.
    # Generally allow NN to be interchangeable with NO3, adding NO2
    # where data present.

    # Specifically:
    # - where NN is missing but we have NO3, set NN to NO3+NO2
    NO2_or_0=tracy['NO2 mg/L N'].copy()
    NO2_or_0.loc[ NO2_or_0.isnull() ]=0.0

    NNmissing=tracy['N+N mg/L N'].isnull()
    tracy.loc[NNmissing,'N+N mg/L N'] = tracy.loc[NNmissing,'NO3 mg/L N'] + NO2_or_0

    # - where NO3 is missing, fill with NN, less NO2 if available, but non-negative
    NO3missing=tracy['NO3 mg/L N'].isnull()
    tracy.loc[NO3missing,'NO3 mg/L N'] = np.clip(tracy.loc[NO3missing,'N+N mg/L N'] - NO2_or_0,0,np.inf)

    # and a few NN samples which are oddly low, and way below a concurrent NO3 sample
    tracy['N+N mg/L N'] = np.maximum(tracy['N+N mg/L N'], tracy['NO3 mg/L N'])

    if 1: # compare NN, NO3, NO2 
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        ax.plot(tracy.Date,tracy['N+N mg/L N'],marker='x',label='NN: Tracy_5B390108001_Nitrite+Nitrate_as_N.csv')
        ax.plot(tracy.Date,tracy['NO2 mg/L N'],marker='x',label='NO2: Tracy_5B390108001_Nitrite_Total_as_N.csv')
        ax.plot(tracy.Date,tracy['NO3 mg/L N'],marker='o',mfc='none',label='NO3: Tracy_5B390108001_Nitrate_Total_as_N.csv')
        ax.legend()
        fig.savefig(os.path.join(outpath,"figures/Tracy-NOx-compare_sources.png"))

    ##

    # Phosphorus:
    if 1:
        # compare P, PO4, TP.
        # Of course TP is the highest, but it is much higher than either of P, PO4
        # There are also numerous samples with coincident P and PO4, with P being maybe
        # twice PO4.
        # These appear to be measuring truly different things, and there is nothing
        # to be gained by any combinations.  Leave as is.

        # Hmm - tracyA2['P mg/L P'] is from "Tracy_5B390108001_Phosphorus_Total_as_P.csv"
        # so better named as 'TP mg/L P', which will then be TP mg/L P_x
        # And PO4 stays as a constant ratio times that CWIQS value
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        ax.plot(tracy.Date,tracy['TP mg/L P_x'],marker='x',  label='Tracy_5B390108001_Phosphorus_Total_as_P.csv')
        ax.plot(tracy.Date,tracy['TP mg/L P_y'],marker='x', label='TP:tracy.csv')
        ax.plot(tracy.Date,tracy['PO4 mg/L P'],marker='x',label='PO4~%.2f*Tracy_5B390108001_Phosphorus_Total_as_P.csv'%prat)
        ax.legend()
        fig.savefig(os.path.join(outpath,"figures/Tracy-P-compare_sources.png"))

    ##

    # Flow:
    tracy['flow mgd']=tracy['flow mgd_x']

    if 1:
        # Hmm - these appear to be very different datasets, too.
        # Emma suggests that the metadata for CWIQS is more reliable than
        # the older dataset, so we should use flow_mgd_x, and discard flow_mgd_y
        # permitted flow is capped at 16 mgd, so no help there in understanding
        # which is correct.
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        # This is the CWIQS data
        ax.plot(tracy.Date,tracy['flow mgd_x'],marker='x',label='delta_sources/Tracy_5B390108001_Flow.csv')
        # This is from Emily's older data stash
        ax.plot(tracy.Date,tracy['flow mgd_y'],marker='x',label='delta_sources/tracy.csv')
        ax.plot(tracy.Date,tracy['flow mgd'],marker='o',lw=0.8,color='k',mfc='none',label='output')
        ax.set_ylabel('flow mgd')
        ax.grid(1)
        ax.legend()
        fig.savefig(os.path.join(outpath,"figures/Tracy-flow-compare_sources.png"))

    # output the final columns:
    tracy_final=tracy.loc[:, ['Date','NH3 mg/L N_x', 'flow mgd_x', 'NO3 mg/L N_x', 'NO2 mg/L N',
                              'N+N mg/L N', 'P mg/L P', 'PO4 mg/L P', 'flow mgd_y', 'NH3 mg/L N_y',
                              'NO3 mg/L N_y', 'TP mg/L P', 'NH3 mg/L N', 'NO3 mg/L N', 'flow mgd'] ]
    tracy_final.to_csv(outfile)
    

    df=tracy
    # plotting
    date, var = var_fields(df["Date"], df["flow mgd"])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(date, var, '-x', color="cadetblue")
    ax.set_title("Tracy: Flow")
    ax.set_ylabel("MGD")
    fig.savefig(outpath + "figures/Tracy_Flow.png")

    date, var = var_fields(df["Date"], df["NH3 mg/L N"])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(date, var, '-x', color="darkslategray")
    ax.set_title("Tracy: Ammonia")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Ammonia.png")

    date, var = var_fields(df["Date"], df["NO3 mg/L N"])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(date, var, '-x', color="seagreen")
    ax.set_title("Tracy: Nitrate")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Nitrate.png")

    # changed to PO4 - though see comparison plots for all P species
    date, var = var_fields(df["Date"], df["PO4 mg/L P"])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(date, var, '-x', color="indianred")
    ax.set_title("Tracy: PO4 as %.2f * TP"%prat)
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Phosphorus.png")

    plt.close('all')
    

### REGIONAL SAN
filename = inpath + "SacRegional.csv"
outfile = outpath + "sac_regional.csv" # keep the Sac Regional name as it is clearer to have
# at least some geographic reference in the name.
df1 = pd.read_csv(filename,parse_dates=['Date'])
df1.rename(columns={'Flow': 'flow mgd', 'NH4_mgL': 'NH4 mg/L N', 
                   'NO3_mgL': 'NO3 mg/L N', 'NO2_mgL': 'NO2 mg/L N',
                   'TKN_mgL': 'TKN mg/L N', 'TP_mgL': 'TP mg/L P', 
                   'PO4_mgL': 'PO4 mg/L P'}, inplace=True)
header = ["Date", "flow mgd", "NH4 mg/L N", "NO3 mg/L N", "NO2 mg/L N", "TP mg/L P", "PO4 mg/L P", "TKN mg/L N"]
df1['Date']=df1.Date.values.astype('M8[D]') # truncate to days
#df1.to_csv(outfile, columns=header)
filename = inpath + "RegionalSan_2012-2017.xlsx"
df2 = pd.ExcelFile(filename).parse('Data')
df2 = data_fill('Ammonia as N, mg/L (Final Eff COMP)', 'Ammonia as N, mg/L (Final Eff GRAB)', df2)
df2 = data_fill('Nitrate as N, mg/L (Final Eff COMP)', 'Nitrate as N, mg/L (Final Eff GRAB)', df2)
df2 = data_fill('Nitrite as N, mg/L (Final Eff COMP)', 'Nitrite as N, mg/L (Final Eff GRAB)', df2)
df2 = data_fill('TKN, mg/l (Final Eff COMP)', 'TKN, mg/l (Final Eff GRAB)', df2)
df2.rename(columns={'Final Effluent Flow, mgd   Daily Avg': 'flow mgd', 
                    'Ammonia as N, mg/L (Final Eff COMP)': 'NH3 mg/L N',
                    'Nitrate as N, mg/L (Final Eff COMP)': 'NO3 mg/L N',
                    'Nitrite as N, mg/L (Final Eff COMP)': 'NO2 mg/L N',
                    'TKN, mg/l (Final Eff COMP)': 'TKN mg/L N',
                    'Total Phosphorus (as P), mg/l (Final Eff COMP)': 'TP mg/L P',
                    'Phosphate, Diss Ortho (as P) (Final Eff COMP)': 'PO4 mg/L P'}, inplace=True)

header = ["Date", "flow mgd", "NH4 mg/L N", "NH3 mg/L N", "NO3 mg/L N", "NO2 mg/L N", 
          "TKN mg/L N", "TP mg/L P", "PO4 mg/L P"]
df = pd.concat([df1, df2])
df.to_csv(outfile, columns=header)


# plotting
date, var = var_fields(df["Date"], df["flow mgd"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="cadetblue")
ax.set_title("Sac Regional: Flow")
ax.set_ylabel("MGD")
fig.savefig(outpath + "figures/RegionalSan_Flow.png")

date, var = var_fields(df["Date"], df["NH4 mg/L N"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="darkslategray")
ax.set_title("Sac Regional: Ammonium")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/RegionalSan_Ammonium.png")

date, var = var_fields(df["Date"], df["NO3 mg/L N"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="seagreen")
ax.set_title("Sac Regional: Nitrate")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/RegionalSan_Nitrate.png")

date, var = var_fields(df["Date"], df["NO2 mg/L N"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="royalblue")
ax.set_title("Sac Regional: Nitrite")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/RegionalSan_Nitrite.png")

date, var = var_fields(df["Date"], df["TP mg/L P"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="indianred")
ax.set_title("Sac Regional: Total Phosphorus")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/RegionalSan_Phosphorus.png")

plt.close('all') # helps with memory use

### STOCKTON
filename = inpath + "Stockton Effluent 1992-09 through 2009-03.csv"
outfile = outpath + "stockton.csv"
df = pd.read_csv(filename,parse_dates=['Date'])
# about 600 lines at the end are missing dates - while we
# could assume that measurements continue daily, but that's
# a guess.  play it safe and drop those:
df=df[ ~df.Date.isnull() ]
df.rename(columns={'Flow_MGD': 'flow mgd', 'NH3_mgL': 'NH3 mg/L N', 
                   'NO3_mgL': 'NO3 mg/L N', 'NO2_mgL': 'NO2 mg/L N',
                   'TKN_mgL': 'TKN mg/L N', 'TP_mgL': 'TP mg/L P'}, inplace=True)
header = ["Date", "flow mgd", "NH3 mg/L N", "NO3 mg/L N", "NO2 mg/L N", "TP mg/L P", "TKN mg/L N"]
df.to_csv(outfile, columns=header)
# plotting
date, var = var_fields(df["Date"], df["flow mgd"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="cadetblue")
ax.set_title("Stockton: Flow")
ax.set_ylabel("MGD")
fig.savefig(outpath + "figures/Stockton_Flow.png")

date, var = var_fields(df["Date"], df["NH3 mg/L N"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="darkslategray")
ax.set_title("Stockton: Ammonia")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Stockton_Ammonia.png")

date, var = var_fields(df["Date"], df["NO3 mg/L N"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="seagreen")
ax.set_title("Stockton: Nitrate")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Stockton_Nitrate.png")

date, var = var_fields(df["Date"], df["NO2 mg/L N"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="royalblue")
ax.set_title("Stockton: Nitrite")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Stockton_Nitrite.png")

#date, var = var_fields(df["Date"], df["TP mg/L P"])
#fig, ax = plt.subplots(figsize=(8,4))
#ax.plot(date, var, '-x', color="indianred")
#ax.set_title("Stockton: Total Phosphorus")
#ax.set_ylabel("mg/L")
#fig.savefig(outpath + "figures/Stockton_Phosphorus.png")

plt.close('all') # helps with memory use

### MANTECA NEW FILE -- this is the one actually used for now.
filename = inpath + "Manteca_2000_2017.csv"
outfile = outpath + "manteca.csv"
# two rows of metadata, one row of column names, one row of units which we skip
# then the data.
df = pd.read_csv(filename, header=2,skiprows=[3],parse_dates=['Date'])
df.rename(columns={'Flow': 'flow mgd', 'Ammonia ': 'NH3 mg/L N', 
                   'Nitrate': 'NO3 mg/L N', 'Nitrite': 'NO2 mg/L N'}, inplace=True)
# RH: setting missing flow to zeros is probably not correct.  This is also a
# place where pandas requires .loc[..,..] style assignment.
# df["flow mgd"][np.isnan(df["flow mgd"])] = 0 
# there are a few, very few, times when there is, e.g., nitrate but no flow, so
# leave in nan flows.  Note that in most cases where flows are missing, the
# remaining analytes are missing, too.
header = ["Date", "flow mgd", "NH3 mg/L N", "NO3 mg/L N", "NO2 mg/L N"]
df.to_csv(outfile, columns=header)

