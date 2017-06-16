import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr

def load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname):
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
	df.to_csv(outfile)
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
		date.append(time[d])
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
am_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)
ind = day_ind(am_df.Time)
time, am = day_avg(am_df.Time, am_df["NH3 mg/L N"], ind)

# flow
filename = inpath + "Davis_5A570100001_Flow.csv"
outfile = outpath + "Davis_Flow.csv"
units = "MGD"
vname = "flow mgd"
flw_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrate
filename = inpath + "Davis_5A570100001_Nitrate_Total_as_N.csv"
outfile = outpath + "Davis_Nitrate.csv"
units = "mg/L"
vname = "NO3 mg/L N"
na_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrite
filename = inpath + "Davis_5A570100001_Nitrite_Total_as_N.csv"
outfile = outpath + "Davis_Nitrite.csv"
units = "mg/L"
vname = "NO2 mg/L N"
ni_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrite + nitrate
csv = path + "Davis_5A570100001_Nitrite+Nitrate_as_N.csv"
outfile = outpath + "Davis_Nitrite+Nitrate.csv"
units = "mg/L"
vname = "N+N mg/L N"
nn_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# phosphorus
csv = path + "Davis_5A570100001_Phosphorus_Total_as_P.csv"
outfile = outpath + "Davis_Phosphorus.csv"
units = "mg/L"
vname = "P mg/L P"
p_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)


### MANTECA
# ammonia
filename = inpath + "Manteca_5B390104001_Ammonia_Total_as_N.csv"
outfile = outpath + "Manteca_Ammonia.csv"
units = "mg/L"
vname = "NH3 mg/L N"
am_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# flow
filename = inpath + "Manteca_5B390104001_Flow.csv"
outfile = outpath + "Manteca_Flow.csv"
units = "MGD"
vname = "flow mgd"
flw_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrate
filename = inpath + "Manteca_5B390104001_Nitrate_Total_as_N.csv"
outfile = outpath + "Manteca_Nitrate.csv"
units = "mg/L"
vname = "NO3 mg/L N"
na_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrite
filename = inpath + "Manteca_5B390104001_Nitrite_Total_as_N.csv"
outfile = outpath + "Manteca_Nitrite.csv"
units = "mg/L"
vname = "NO2 mg/L N"
ni_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrite + nitrate
csv = path + "Manteca_5B390104001_Nitrite+Nitrate_as_N.csv"
outfile = outpath + "Manteca_Nitrite+Nitrate.csv"
units = "mg/L"
vname = "N+N mg/L N"
nn_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrogen
csv = path + "Manteca_5B390104001_Nitrogen_as_N.csv"
outfile = outpath + "Manteca_Nitrogen.csv"
units = "mg/L"
vname = "N mg/L N"
nn_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

### TRACY
# ammonia
filename = inpath + "Tracy_5B390108001_Ammonia_Total_as_N.csv"
outfile = outpath + "Tracy_Ammonia.csv"
units = "mg/L"
vname = "NH3 mg/L N"
am_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# flow
filename = inpath + "Tracy_5B390108001_Flow.csv"
outfile = outpath + "Tracy_Flow.csv"
units = "MGD"
vname = "flow mgd"
flw_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrate
filename = inpath + "Tracy_5B390108001_Nitrate_Total_as_N.csv"
outfile = outpath + "Tracy_Nitrate.csv"
units = "mg/L"
vname = "NO3 mg/L N"
na_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrite
filename = inpath + "Tracy_5B390108001_Nitrite_Total_as_N.csv"
outfile = outpath + "Tracy_Nitrite.csv"
units = "mg/L"
vname = "NO2 mg/L N"
ni_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# nitrite + nitrate
csv = path + "Tracy_5B390108001_Nitrite+Nitrate_as_N.csv"
outfile = outpath + "Tracy_Nitrite+Nitrate.csv"
units = "mg/L"
vname = "N+N mg/L N"
nn_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

# phosphorus
csv = path + "Tracy_5B390108001_Phosphorus_Total_as_P.csv"
outfile = outpath + "Tracy_Phosphorus.csv"
units = "mg/L"
vname = "P mg/L P"
p_df = load_wwtp(filename, outfile, tname, tformat, varname, unitsname, units, method, cols, vname)

