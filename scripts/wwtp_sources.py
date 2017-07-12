import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr

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
    
    
# variables that don't change between wwtp files
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

# save final concatenated file
davis.to_csv(outpath+"davis.csv")

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

### TRACY
if 0:
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
    ax.set_title("Tracy: Total Phosphorus (as P)")
    ax.set_ylabel("mg/L")
    fig.savefig(outpath + "figures/Tracy_Phosphorus.png")    
    
    # concatenate
    tracy = pd.merge(tracy, df6, how='outer', on='Date')
    
    # save final concatenated file
    tracy.to_csv(outpath+"tracy.csv")
    
def var_fields(date, variable):
    time = []
    var = []
    for i in range(len(date)):
        if np.isnan(float(variable[i]))==False:
            time.append(dt.datetime.strptime(date[i], "%m/%d/%Y"))
            var.append(variable[i]) 
    return time, np.asarray(var)

### REGIONAL SAN
filename = inpath + "SacRegional.csv"
outfile = outpath + "sac_regional.csv" # keep the Sac Regional name as it is clearer to have
# at least some geographic reference in the name.
df = pd.read_csv(filename)
df.rename(columns={'Flow': 'flow mgd', 'NH4_mgL': 'NH4 mg/L N', 
                   'NO3_mgL': 'NO3 mg/L N', 'NO2_mgL': 'NO2 mg/L N',
                   'TKN_mgL': 'TKN mg/L N', 'TP_mgL': 'TP mg/L P', 
                   'PO4_mgL': 'PO4 mg/L P'}, inplace=True)
header = ["Date", "flow mgd", "NH3 mg/L N", "NO3 mg/L N", "NO2 mg/L N", "TP mg/L P", "PO4 mg/L P", "TKN mg/L N"]
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


### STOCKTON
filename = inpath + "Stockton Effluent 1992-09 through 2009-03.csv"
outfile = outpath + "stockton.csv"
df = pd.read_csv(filename)
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

### TRACY
filename = inpath + "tracy.csv"
outfile = outpath + "tracy.csv"
df = pd.read_csv(filename)
df.rename(columns={'Flow': 'flow mgd', 'NH3': 'NH3 mg/L N', 
                   'NO3': 'NO3 mg/L N', 'TP': 'TP mg/L P'}, inplace=True)
df.to_csv(outfile)
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

date, var = var_fields(df["Date"], df["TP mg/L P"])
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(date, var, '-x', color="indianred")
ax.set_title("Tracy: Total Phosphorus")
ax.set_ylabel("mg/L")
fig.savefig(outpath + "figures/Tracy_Phosphorus.png")

### MANTECA NEW FILE -- this is the one actually used for now.
filename = inpath + "Manteca_2000_2017.csv"
outfile = outpath + "manteca.csv"
df = pd.read_csv(filename, header=3)
df.rename(columns={'Unnamed: 0': 'Date', 'MGD': 'flow mgd', 'mg/L as N.2': 'NH3 mg/L N', 
                   'mg/L as N': 'NO3 mg/L N', 'mg/L as N.1': 'NO2 mg/L N', 'Date': 'Time'}, inplace=True)
df["flow mgd"][np.isnan(df["flow mgd"])] = 0
header = ["Date", "flow mgd", "NH3 mg/L N", "NO3 mg/L N", "NO2 mg/L N"]
df.to_csv(outfile, columns=header)



