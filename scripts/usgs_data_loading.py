import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


def load_usgs(filename, header, tformat, dname, tname, tzname, varname):
	dat = pd.read_csv(filename, header=header, delim_whitespace=True)
	dat[dname+tname] = dat[dname] + " " + dat[tname]
	dtime = [dt.datetime.strptime(dat[dname+tname][i], tformat) \
			 for i in range(1, len(dat[dname+tname][:]))]
	for i in range(len(dtime)):
		if dat[tzname][i+1] == 'PST':
			dtime[i] += dt.timedelta(hours=8)
		elif dat[tzname][i+1] == 'PDT':
			dtime[i] += dt.timedelta(hours=7)
		dtime[i].replace(tzinfo=dt.timezone.utc)
	var = dat[varname][1:]
	times = [dtime[i].timestamp() for i in range(len(dtime))]
	return dtime, times, var
	
def day_ind(dtime):
	d = 0 
	ind = np.zeros(len(dtime))
	for i in range(1,len(dtime)):
		if dtime[i].date() == dtime[i-1].date():
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
		date.append(time[d].date())
		dvar[i] = np.mean(var[np.where(ind==i)[0]].astype(np.float))
		d += len(np.where(ind==i)[0])
	return date, dvar
	
	
inpath = "../sources/delta_sources/"	
tformat = "%Y-%m-%d %H:%M"
dname = "date"
tname = "time"
tzname = "tz_cd"
	
##### VERNALIS : DISCHARGE
filename = inpath + "USGS_11303500_Vernalis_Discharge.txt"
header = 29
varname = "15169_00060"
dtime, times, discharge = load_usgs(filename, header, tformat, dname, tname, tzname, varname)
ind = day_ind(dtime)
dates, ddis = day_avg(dtime, discharge, ind)
dat_new = {'Date': dates,
		   'flow cfs': ddis}
df1 = pd.DataFrame(dat_new, columns=['Date','flow cfs'])		   
#df.to_csv("../outputs/intermediate/delta/SanJoaquinDischarge.csv")
		   
#plt.figure()
#plt.plot_date(dtime, discharge, '-')	
#plt.title(filename)

##### VERNALIS : NUTRIENTS	
filename = inpath + "USGS_11303500_Vernalis_Nutrients.txt"
header = 26
varname = "15171_99133"
dtime, times, nutrients = load_usgs(filename, header, tformat, dname, tname, tzname, varname)
ind = day_ind(dtime)
dates, dnut = day_avg(dtime, nutrients, ind)
dat_new = {'Date': dates,
		   'N+N mg/L N': dnut}
df2 = pd.DataFrame(dat_new, columns=['Date','N+N mg/L N'])		   
#df.to_csv("../outputs/intermediate/delta/SanJoaquinNutrients.csv")

# concatenate and save
sanj = pd.merge(df1, df2, how='outer', on='Date')
sanj.to_csv("../outputs/intermediate/delta/SanJoaquin.csv")	
	
#plt.figure()	
#plt.plot_date(ntime, nutrients, '-')	
#plt.title(filename)
	
	
##### FREEPORT : DISCHARGE	
filename = inpath + "USGS_11447650_Freeport_Discharge.txt"
header = 28
varname = "176626_00060"
dtime, times, discharge = load_usgs(filename, header, tformat, dname, tname, tzname, varname)
ind = day_ind(dtime)
dates, ddis = day_avg(dtime, discharge, ind)
dat_new = {'Date': dates,
		   'flow cfs': ddis}
df1 = pd.DataFrame(dat_new, columns=['Date','flow cfs'])		   
#df1.to_csv("../outputs/intermediate/delta/SacramentoFreeportDischarge.csv")
	
#plt.figure()
#plt.plot_date(dtime, discharge, '-')
#plt.title(filename)
	
##### FREEPORT : NUTRIENTS
filename = inpath + "USGS_11447650_Freeport_Nutrients.txt"
header = 26
varname = "15759_99133"
dtime, times, nutrients = load_usgs(filename, header, tformat, dname, tname, tzname, varname)
ind = day_ind(dtime)
dates, dnut = day_avg(dtime, nutrients, ind)
dat_new = {'Date': dates,
		   'N+N mg/L N': dnut}
df2 = pd.DataFrame(dat_new, columns=['Date','N+N mg/L N'])		   
#df2.to_csv("../outputs/intermediate/delta/SacramentoFreeportNutrients.csv")
	
#plt.figure()
#plt.plot_date(ntime, nutrients, '-')
#plt.title(filename)
		
##### VERONA : DISCHARGE		
filename = inpath + "USGS_11425500_Verona_Discharge.txt"
header = 30
varname = "15690_00060"
dtime, times, discharge = load_usgs(filename, header, tformat, dname, tname, tzname, varname)
ind = day_ind(dtime)
dates, ddis = day_avg(dtime, discharge, ind)
dat_new = {'Date': dates,
		   'flow cfs': ddis}
df3 = pd.DataFrame(dat_new, columns=['Date','flow cfs'])		   
#df3.to_csv("../outputs/intermediate/delta/SacramentoVeronaDischarge.csv")
	
#plt.figure()
#plt.plot_date(dtime, discharge, '-')
#plt.title(filename)

# concatenate and save freeport nutrients and verona discharge *** check about what data to use
sac = pd.merge(df2, df3, how='outer', on='Date')
sac.to_csv("../outputs/intermediate/delta/Sacramento.csv")