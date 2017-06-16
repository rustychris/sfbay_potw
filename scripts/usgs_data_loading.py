import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

##### VERNALIS : DISCHARGE
filename = "USGS_11303500_Vernalis_Discharge.txt"
dat = pd.read_csv(filename, header=29, delim_whitespace=True)
dat["datetime"] = dat["date"] + " " + dat["time"]

dtime = []
for i in range(1,len(dat["datetime"][:])):
	dtime.append(dt.datetime.strptime(dat["datetime"][i], "%Y-%m-%d %H:%M"))
for i in range(len(dtime)):
	if dat["tz_cd"][i+1] == 'PST':
		dtime[i] += dt.timedelta(hours=8)
	elif dat["tz_cd"][i+1] == 'PDT':
		dtime[i] += dt.timedelta(hours=7)
	dtime[i].replace(tzinfo=dt.timezone.utc)

discharge = dat["15169_00060"][1:]

times = []
for i in range(len(dtime)):
	times.append(dtime[i].timestamp)

dat_new = {'time': times,
		   'flow': discharge}
#df = pd.DataFrame(dat_new, columns=['time','flow'])		   
#dat_new.to_csv("name.csv")
		   
plt.figure()
plt.plot_date(dtime, discharge, '-')	
plt.title(filename)

##### VERNALIS : NUTRIENTS	
filename = "USGS_11303500_Vernalis_Nutrients.txt"
dat = pd.read_csv(filename, header=26, delim_whitespace=True)
dat["datetime"] = dat["date"] + " " + dat["time"]

ntime = []
for i in range(1,len(dat["datetime"][:])):
	ntime.append(dt.datetime.strptime(dat["datetime"][i], "%Y-%m-%d %H:%M"))
for i in range(len(ntime)):
	if dat["tz_cd"][i+1] == 'PST':
		ntime[i] += dt.timedelta(hours=8)
	elif dat["tz_cd"][i+1] == 'PDT':
		ntime[i] += dt.timedelta(hours=7)
	ntime[i].replace(tzinfo=dt.timezone.utc)
	
nutrients = dat["15171_99133"][1:]
times = []
for i in range(len(ntime)):
	times.append(ntime[i].timestamp)

dat_new = {'time': times,
		   'nutrients': nutrients}	
	
plt.figure()	
plt.plot_date(ntime, nutrients, '-')	
plt.title(filename)
	
	
##### FREEPORT : DISCHARGE	
filename = "USGS_11447650_Freeport_Discharge.txt"
dat = pd.read_csv(filename, header=28, delim_whitespace=True)
dat["datetime"] = dat["date"] + " " + dat["time"]

dtime = []
for i in range(1,len(dat["datetime"][:])):
	dtime.append(dt.datetime.strptime(dat["datetime"][i], "%Y-%m-%d %H:%M"))
for i in range(len(ntime)):
	if dat["tz_cd"][i+1] == 'PST':
		dtime[i] += dt.timedelta(hours=8)
	elif dat["tz_cd"][i+1] == 'PDT':
		dtime[i] += dt.timedelta(hours=7)
	dtime[i].replace(tzinfo=dt.timezone.utc)
	
discharge = dat["176626_00060"][1:]
times = []
for i in range(len(dtime)):
	times.append(dtime[i].timestamp)

dat_new = {'time': times,
		   'flow': discharge}
	
plt.figure()
plt.plot_date(dtime, discharge, '-')
plt.title(filename)
	
##### FREEPORT : NUTRIENTS
filename = "USGS_11447650_Freeport_Nutrients.txt"
dat = pd.read_csv(filename, header=26, delim_whitespace=True)
dat["datetime"] = dat["date"] + " " + dat["time"]

ntime = []
for i in range(1,len(dat["datetime"][:])):
	ntime.append(dt.datetime.strptime(dat["datetime"][i], "%Y-%m-%d %H:%M"))
for i in range(len(ntime)):
	if dat["tz_cd"][i+1] == 'PST':
		ntime[i] += dt.timedelta(hours=8)
	elif dat["tz_cd"][i+1] == 'PDT':
		ntime[i] += dt.timedelta(hours=7)
	ntime[i].replace(tzinfo=dt.timezone.utc)
	
nutrients = dat["15759_99133"][1:]
times = []
for i in range(len(ntime)):
	times.append(ntime[i].timestamp)

dat_new = {'time': times,
		   'nutrients': nutrients}	
	
plt.figure()
plt.plot_date(ntime, nutrients, '-')
plt.title(filename)
		
##### VERONA : DISCHARGE		
filename = "USGS_11425500_Verona_Discharge.txt"
dat = pd.read_csv(filename, header=30, delim_whitespace=True)
dat["datetime"] = dat["date"] + " " + dat["time"]

dtime = []
for i in range(1,len(dat["datetime"][:])):
	dtime.append(dt.datetime.strptime(dat["datetime"][i], "%Y-%m-%d %H:%M"))
for i in range(len(ntime)):
	if dat["tz_cd"][i+1] == 'PST':
		dtime[i] += dt.timedelta(hours=8)
	elif dat["tz_cd"][i+1] == 'PDT':
		dtime[i] += dt.timedelta(hours=7)
	dtime[i].replace(tzinfo=dt.timezone.utc)
	
discharge = dat["15690_00060"][1:]
times = []
for i in range(len(dtime)):
	times.append(dtime[i].timestamp)

dat_new = {'time': times,
		   'flow': discharge}
	
plt.figure()
plt.plot_date(dtime, discharge, '-')
plt.title(filename)