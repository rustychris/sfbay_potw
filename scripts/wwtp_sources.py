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
	dvar = np.zeros(ind[-1])
	d = 0 
	for i in range(int(ind[-1])):
		date.append(time[d].to_datetime().date())
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

### DAVIS
# ammonia
filename = inpath + "Davis_5A570100001_Ammonia_Total_as_N.csv"
outfile = outpath + "Davis_Ammonia.csv"
units = "mg/L"
vname = "NH3 mg/L N"
am_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(am_df.Time)
time, am = day_avg(am_df.Time, am_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(am)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# flow
filename = inpath + "Davis_5A570100001_Flow.csv"
outfile = outpath + "Davis_Flow.csv"
units = "MGD"
vname = "flow mgd"
flw_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(flw_df.Time)
time, flw = day_avg(flw_df.Time, flw_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(flw)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrate
filename = inpath + "Davis_5A570100001_Nitrate_Total_as_N.csv"
outfile = outpath + "Davis_Nitrate.csv"
units = "mg/L"
vname = "NO3 mg/L N"
na_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(na_df.Time)
time, na = day_avg(na_df.Time, na_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(na)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrite
filename = inpath + "Davis_5A570100001_Nitrite_Total_as_N.csv"
outfile = outpath + "Davis_Nitrite.csv"
units = "mg/L"
vname = "NO2 mg/L N"
ni_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(ni_df.Time)
time, ni = day_avg(ni_df.Time, ni_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(ni)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrite + nitrate
filename = inpath + "Davis_5A570100001_Nitrite+Nitrate_as_N.csv"
outfile = outpath + "Davis_Nitrite+Nitrate.csv"
units = "mg/L"
vname = "N+N mg/L N"
nn_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(nn_df.Time)
time, nn = day_avg(nn_df.Time, nn_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(nn)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# phosphorus
filename = inpath + "Davis_5A570100001_Phosphorus_Total_as_P.csv"
outfile = outpath + "Davis_Phosphorus.csv"
units = "mg/L"
vname = "P mg/L P"
p_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(p_df.Time)
time, p = day_avg(p_df.Time, p_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(p),
	 'PO4 mg/L P': pd.Series(p/2.23)}
df = pd.DataFrame(d)
df.to_csv(outfile)


### MANTECA
# ammonia
filename = inpath + "Manteca_5B390104001_Ammonia_Total_as_N.csv"
outfile = outpath + "Manteca_Ammonia.csv"
units = "mg/L"
vname = "NH3 mg/L N"
am_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(am_df.Time)
time, am = day_avg(am_df.Time, am_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(am)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# flow
filename = inpath + "Manteca_5B390104001_Flow.csv"
outfile = outpath + "Manteca_Flow.csv"
units = "MGD"
vname = "flow mgd"
flw_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(flw_df.Time)
time, flw = day_avg(flw_df.Time, flw_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(flw)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrate
filename = inpath + "Manteca_5B390104001_Nitrate_Total_as_N.csv"
outfile = outpath + "Manteca_Nitrate.csv"
units = "mg/L"
vname = "NO3 mg/L N"
na_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(na_df.Time)
time, na = day_avg(na_df.Time, na_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(na)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrite
filename = inpath + "Manteca_5B390104001_Nitrite_Total_as_N.csv"
outfile = outpath + "Manteca_Nitrite.csv"
units = "mg/L"
vname = "NO2 mg/L N"
ni_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(ni_df.Time)
time, ni = day_avg(ni_df.Time, ni_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(ni)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrite + nitrate
filename = inpath + "Manteca_5B390104001_Nitrite+Nitrate_as_N.csv"
outfile = outpath + "Manteca_Nitrite+Nitrate.csv"
units = "mg/L"
vname = "N+N mg/L N"
nn_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(nn_df.Time)
time, nn = day_avg(nn_df.Time, nn_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(nn)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrogen
filename = inpath + "Manteca_5B390104001_Nitrogen_Total_as_N.csv"
outfile = outpath + "Manteca_Nitrogen.csv"
units = "mg/L"
vname = "N mg/L N"
n_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(n_df.Time)
time, n = day_avg(n_df.Time, n_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(n)}
df = pd.DataFrame(d)
df.to_csv(outfile)


### TRACY
# ammonia
filename = inpath + "Tracy_5B390108001_Ammonia_Total_as_N.csv"
outfile = outpath + "Tracy_Ammonia.csv"
units = "mg/L"
vname = "NH3 mg/L N"
am_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(am_df.Time)
time, am = day_avg(am_df.Time, am_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(am)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# flow
filename = inpath + "Tracy_5B390108001_Flow.csv"
outfile = outpath + "Tracy_Flow.csv"
units = "MGD"
vname = "flow mgd"
flw_df = load_wwtp(filename=filename, tname=tname, tformat=tformat, varname=varname, unitsname=unitsname, units=units, method="Analytical Method", cols=[2,5,6,12], vname=vname)
ind = day_ind(flw_df.Time)
time, flw = day_avg(flw_df.Time, flw_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(flw)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrate
filename = inpath + "Tracy_5B390108001_Nitrate_Total_as_N.csv"
outfile = outpath + "Tracy_Nitrate.csv"
units = "mg/L"
vname = "NO3 mg/L N"
na_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(na_df.Time)
time, na = day_avg(na_df.Time, na_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(na)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrite
filename = inpath + "Tracy_5B390108001_Nitrite_Total_as_N.csv"
outfile = outpath + "Tracy_Nitrite.csv"
units = "mg/L"
vname = "NO2 mg/L N"
ni_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(ni_df.Time)
time, ni = day_avg(ni_df.Time, ni_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(ni)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# nitrite + nitrate
filename = inpath + "Tracy_5B390108001_Nitrite+Nitrate_as_N.csv"
outfile = outpath + "Tracy_Nitrite+Nitrate.csv"
units = "mg/L"
vname = "N+N mg/L N"
nn_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(nn_df.Time)
time, nn = day_avg(nn_df.Time, nn_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(nn)}
df = pd.DataFrame(d)
df.to_csv(outfile)


# phosphorus
filename = inpath + "Tracy_5B390108001_Phosphorus_Total_as_P.csv"
outfile = outpath + "Tracy_Phosphorus.csv"
units = "mg/L"
vname = "P mg/L P"
p_df = load_wwtp(filename, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(p_df.Time)
time, p = day_avg(p_df.Time, p_df[vname], ind)
d = {'Time': pd.Series(time),
	 vname: pd.Series(p),
	 'PO4 mg/L P': pd.Series(p/2.23)}
df = pd.DataFrame(d)
df.to_csv(outfile)
