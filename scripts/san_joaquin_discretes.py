"""
Created on Wed Nov 22 15:19:46 2017
# ~zhenlin/dwaq/cascade/suisun_nutrient_cycling/sanJoaquinLoad.py
@author: zhenlinz

This file relies on a separate compilation of nutrient observations
which is too large to include in the github repository.  The
output of the script is 
../outputs/intermediate/delta/san_joaquin_composite_conc_zz.csv
which *is* part of the repository.
"""

import xarray as xr
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import os
import numpy as np
from dateutil import rrule
from datetime import date
import math

##  load the source NH3 and NO3 concentration data at CA10 and USGS station

databasepath = "../sources/delta_sources"
databasefile = os.path.join(databasepath,"DeltaSuisun2.sqlite")
conn = sqlite3.connect(databasefile)
df = pd.read_sql_query("""
                       	SELECT 
		                   "STATION NUMBER",   
                          "Collection Date",
                          "Result",
                          "Analyte",
                          "Units",
		                   StationTable.Latitude AS Lat,
		                   StationTable.Longitude AS Lon,
                          StationTable."Site Name" AS SiteName
	                    FROM 
		                   DiscreteTable
	                    INNER JOIN StationTable on DiscreteTable."Station Number" = StationTable."Site Number"
	                    WHERE (Analyte="Dissolved Nitrate + Nitrite"  
                        OR Analyte="Dissolved Ammonia"
                        OR Analyte="Dissolved Silica (SiO2)"
                        OR Analyte="Dissolved Ortho-phosphate")
                        AND julianday("Collection Date")>=julianday('2000-01-01')
                        AND julianday("Collection Date")<julianday('2016-12-31')		                  
                        ORDER BY Lat;
                       """,
                       conn)

conn.close()

df['Result'] = df['Result'].replace('< R.L.',0) # < R.L.: below reporting limit
df['Result'] = df['Result'].replace('N.A.',np.nan) # 'N.A.': not analyzed, so no data is available.  
df['Result'] = df['Result'].replace('N.S.',np.nan)
df['Result'] = df['Result'].replace('',np.nan)

#%%
C10 = df[df['Station Number']=='B9D74051159']
USGS = df[df['Station Number']=='USGS-11303500']
M6 = df[df['Station Number'] == 'B0D74781185']
M2 = df[df['Station Number'] == 'B0D74821187']
RS = df[df['Station Number'] == 'B0D74831187']
OR = df[df['Station Number'] == 'B9D74851200']

C10_NOx = C10[C10['Analyte']=='Dissolved Nitrate + Nitrite']
C10_NH4 = C10[C10['Analyte']=='Dissolved Ammonia']
C10_SiO2 = C10[C10['Analyte']=='Dissolved Silica (SiO2)']
C10_PO4 = C10[C10['Analyte'] == 'Dissolved Ortho-phosphate']

USGS_NOx = USGS[USGS['Analyte']=='Dissolved Nitrate + Nitrite']
USGS_NH4 = USGS[USGS['Analyte']=='Dissolved Ammonia']
USGS_SiO2 = USGS[USGS['Analyte']=='Dissolved Silica (SiO2)']
USGS_PO4 = USGS[USGS['Analyte'] == 'Dissolved Ortho-phosphate']

M6_NOx = M6[M6['Analyte']=='Dissolved Nitrate + Nitrite']
M6_NH4 = M6[M6['Analyte']=='Dissolved Ammonia']
M6_PO4 = M6[M6['Analyte'] == 'Dissolved Ortho-phosphate']

M2_NOx = M2[M2['Analyte']=='Dissolved Nitrate + Nitrite']
M2_NH4 = M2[M2['Analyte']=='Dissolved Ammonia']
M2_PO4 = M2[M2['Analyte'] == 'Dissolved Ortho-phosphate']

RS_NOx = RS[RS['Analyte']=='Dissolved Nitrate + Nitrite']
RS_NH4 = RS[RS['Analyte']=='Dissolved Ammonia']
RS_PO4 = RS[RS['Analyte'] == 'Dissolved Ortho-phosphate']

OR_NOx = OR[OR['Analyte']=='Dissolved Nitrate + Nitrite']
OR_NH4 = OR[OR['Analyte']=='Dissolved Ammonia']
OR_PO4 = OR[OR['Analyte'] == 'Dissolved Ortho-phosphate']

#%%
fig, ax = plt.subplots()
plt.plot(pd.to_datetime(C10_NOx['Collection Date']),C10_NOx['Result'],'o',label='C10')
plt.plot(pd.to_datetime(USGS_NOx['Collection Date']),USGS_NOx['Result'],'o',label='USGS')
plt.plot(pd.to_datetime(M6_NOx['Collection Date']),M6_NOx['Result'],'o',label='M6')
plt.plot(pd.to_datetime(M2_NOx['Collection Date']),M2_NOx['Result'],'o',label='M2')
plt.plot(pd.to_datetime(RS_NOx['Collection Date']),RS_NOx['Result'],'o',label='RS')
plt.plot(pd.to_datetime(OR_NOx['Collection Date']),OR_NOx['Result'],'o',label='OR')
plt.legend()

fig, ax = plt.subplots()
plt.plot(pd.to_datetime(C10_NH4['Collection Date']),C10_NH4['Result'],'o',label='C10')
plt.plot(pd.to_datetime(USGS_NH4['Collection Date']),USGS_NH4['Result'],'o',label='USGS')
plt.plot(pd.to_datetime(M6_NH4['Collection Date']),M6_NH4['Result'],'o',label='M6')
plt.plot(pd.to_datetime(M2_NH4['Collection Date']),M2_NH4['Result'],'o',label='M2')
plt.plot(pd.to_datetime(RS_NH4['Collection Date']),RS_NH4['Result'],'o',label='RS')
plt.plot(pd.to_datetime(OR_NH4['Collection Date']),OR_NH4['Result'],'o',label='OR')
plt.legend()

## Using monthly-averaged nutrient values from nearby 6 stations as boundary condition for San Joaquin. 

timemonth = list(rrule.rrule(rrule.MONTHLY,dtstart=date(2000,1,1),until=date(2017,1,1)))
sjr_data = pd.concat([C10,USGS,M6,M2,RS,OR])
sjr_dates = pd.to_datetime(sjr_data['Collection Date'])

nutrientList = ['Dissolved Nitrate + Nitrite',
                'Dissolved Ammonia',
                'Dissolved Silica (SiO2)',
                'Dissolved Ortho-phosphate']
sjr_n = []
for i in range(len(timemonth)-1):
    sjr_r = sjr_data[(sjr_dates>timemonth[i]) & (sjr_dates<=timemonth[i+1])]
    try:
        sjr_nr = [np.nanmean(sjr_r[sjr_r['Analyte']== nutrientListi]['Result'].values.astype(float)) 
                  for nutrientListi in nutrientList]
        sjr_n.append(sjr_nr)
    except RuntimeWarning:
        sjr_n.append([np.nan]*4)

sjr_n.append([np.nan]*4) # append nan to the last time stamp. 
sjr_n = np.asarray(sjr_n)

#%% Now replacing the values in xarray

sjr_df = pd.DataFrame({'time':timemonth,
                       'NO3':sjr_n[:,0],
                       'NH4':sjr_n[:,1],
                       'SiO2':sjr_n[:,2]*60./(60+16*2),
                       'PO4':sjr_n[:,3]})  
sjr_df.set_index(['time'])
sjr_df.index = pd.to_datetime(timemonth)
    
sjr_df_new = sjr_df.resample('1d').ffill()

sjr_df_new.to_csv('../outputs/intermediate/delta/san_joaquin_composite_conc_zz.csv')
