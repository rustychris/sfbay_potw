"""
Compile Delta Data

**Estimates loads entering Suisun Bay from the Delta**

This script ingests DAYFLOW data, and nutrient data from [EMP](http://www.water.ca.gov/bdma/meta/continuous.cfm) (both lab and field datasets).  The output is flow and nutrient time series for Delta inputs to Suisun Bay.

Note that this is intended for the SUNTANS model with false deltas. Flows are supposed 
to roughly correspond to Sacramento/San Joaquin fractions exiting the Delta.
Likewise, nutrients estimates reflect nutrient levels leaving the
Delta.  In the biogeochemical model this is a significant approximation since nutrient transformations
will be applied in the false delta.  Nonetheless, we assume that modeled
transformation in the false delta are small compared to actual
Delta nutrient transformation, so better to take Delta output.

Processing nutrient data follows Emily Novick's *Suisun loads.r* script, as of 2016-02-08.

"""

# Imports
from __future__ import print_function
import os
import utils
import pandas as pd
import numpy as np


# path to POTW data files
sources_path="../sources"
compile_path='../outputs/intermediate'


# In[14]:


dayflow=pd.read_csv(os.path.join(sources_path,'DAYFLOW_1975_2016.csv'),
                    parse_dates=['DATE '])
# remove whitespace around header names
dayflow.rename(columns={v: v.strip() for v in dayflow.columns},inplace=True)
dayflow.rename(columns={'Unnamed: 29':'X2'},inplace=True)
# ends 2015-09-30

# Split Sac and SJ flows, but adjusted for net outflow
# Sum of sac and sj equals out
# but note that west is often negative!
sac_frac=dayflow.RIO[:].astype(np.float64) / (dayflow.RIO[:] + dayflow.WEST[:])
sac=sac_frac * dayflow.OUT[:]
sjr=(1-sac_frac)*dayflow.OUT[:]
reverse=sjr.clip(-np.inf,0)
# rather than allow reverse flows, which would likely throw off the
# salt balance, force sjr to be non-negative
sjr-=reverse
sac+=reverse

# keep these in CFS for easy comparison, but write in cf-compliant way.
df_sac=pd.DataFrame( {'Date':dayflow.DATE,
                      'flow ft3/s':sac} ).set_index('Date')

df_sj=pd.DataFrame( {'Date':dayflow.DATE,
                     'flow ft3/s':sjr} ).set_index('Date')


# In[15]:


# from Suisun loads.r

s=pd.read_csv(os.path.join(sources_path,'sfb_data_2013-08-15_all nutrients.csv'),
              parse_dates=['Date'])

# Consolidate discrete and calculated
s['chl']=np.where(pd.isnull(s.dchl),s.cchl,s.dchl)
s['do'] =np.where(pd.isnull(s.do),s.cdo,s.do)
s['spm']=np.where(pd.isnull(s.dspm),s.cspm,s.dspm)

s.drop(['dchl','cchl','cdo','cspm','dspm'],axis=1,inplace=True)

s['din']=s.nh + s.nn

## Process EMP lab and field data

def merge_stations(emp):
    for a,b in [ ('C10A','C10'),
                 ('C3A', 'C3'),
                 ('P12A', 'P12'),
                 ('P10A', 'P10'),
                 ('MD10A', 'MD10'),
                 ('MD7A', 'MD7')]:
        emp.loc[ emp.StationCode==a, 'StationCode'] = b
    return emp

# if this craps out, might be because Matrix no longer has a space.
field=pd.read_csv(os.path.join(sources_path,'EMP_Field_1975_2012.csv'),
                  parse_dates=['SampleDate'],
                  usecols=['SampleDate','StationCode', 
                           'Depth',' Matrix','AnalyteName','Result',
                           'MethodName','FractionName', 'UnitName'],
                  na_values=['n/p'],
                  dtype={'Depth':np.float64})

field.rename(columns={v: v.strip() for v in field.columns},inplace=True)
merge_stations(field)

# Keep variables of interest and adequately sampled.
field=field[ (field.Matrix=='Water') & 
             (field.AnalyteName.isin( ['Conductance (EC)', 'Oxygen', 'pH', 
                                       'Secchi Depth', 'Temperature', 'Turbidity'] )) ]
# restrict to interesting columns, and have depth and result 
field=field.loc[ ~field.Depth.isnull() & ~field.Result.isnull(),
                 ['SampleDate', 'StationCode', 'Depth', 'AnalyteName', 'Result']]

# Just care about depth-averaged, so average over Depth
# then discard
field=field.groupby(['SampleDate','StationCode','AnalyteName']).mean()['Result'].unstack()
field.head()


# In[16]:


lab0 =pd.read_csv(os.path.join(sources_path,'EMP_Lab_1975_2012.csv'),
                  parse_dates=['SampleDate'],
                  usecols=['StationCode', 'Depth', 'SampleDate','ConstituentName', 
                           'ResultPrefix', 'Result', 'UnitName','ReportingLimit',
                           'Group' ],
                  na_values=['n/p'],
                  dtype={'Depth':np.float64})
merge_stations(lab0)

# Fix ResultPrefixes and estimate censored data
lab0.Result=np.where( lab0.ResultPrefix.isin(['<','< ']),
                      lab0.ReportingLimit/2., 
                      lab0.Result  )
lab1=lab0.loc[ lab0.Group.isin(['Biological', 'Nutrients', 'Other']) &
               ~(lab0.Depth.isnull() | lab0.Result.isnull()) ,
               ['SampleDate', 'StationCode', 'Depth', 'ConstituentName', 'Result']]

lab1=lab1.groupby(['SampleDate','StationCode','ConstituentName']).mean()['Result'].unstack()


# In[17]:


# column renames, coalescing sufficiently similar columns.
lab2=lab1.copy()

nh4_diss=lab2['Ammonia (Dissolved)']
nh4_tot =lab2['Ammonia (Total)']
lab2['NH4 mg/L N']=np.where(nh4_diss.isnull(),nh4_tot,nh4_diss)


lab2.rename(columns={
    'Chlorophyll a':'Chl-a ug/L',
    'Kjeldahl Nitrogen (Total)':'TKN mg/L N',
    'Nitrate (Dissolved)':'NO3 mg/L N',
    'Nitrite (Dissolved)':'NO2 mg/L N',
    'Nitrite + Nitrate (Dissolved)':'NOx mg/L N',
    'Organic Nitrogen (Dissolved)':'DON mg/L N',
    'Organic Nitrogen (Total)':'TON mg/L N',
    'Ortho-phosphate (Dissolved)':'PO4 mg/L P',
    'Phosphorus (Total)':'TP mg/L P', 
    'Silica (SiO2) (Dissolved)':'SiO2 mg/L Si',
    'Solids (Total Dissolved)':'TDS mg/L',
    'Solids (Total Suspended)':'TSS mg/L',
},inplace=True)

lab2['DIN mg/L N']=lab2['NH4 mg/L N']+lab2['NOx mg/L N']

lab2=lab2.loc[:,['Chl-a ug/L', 
                 'TKN mg/L N', 
                 'NO3 mg/L N',
                 'NO2 mg/L N',
                 'NOx mg/L N',
                 'DON mg/L N', 
                 'TON mg/L N', 
                 'PO4 mg/L P',
                 'TP mg/L P', 
                 'SiO2 mg/L Si', 
                 'TDS mg/L',
                 'TSS mg/L',
                 'NH4 mg/L N',
                 'DIN mg/L N']]
lab2.head()


# In[18]:


# At this point data in hand are:
# df_sac, df_sj: flows
# s: Polaris cruises
# field,lab: EMP test data

# pulling NH4, NO3 and PO4 out is a little complicated -
# 3 sources for each
# D24/D16 through 1995 (ignore - before the target analysis period)
# D19 and D24 regression through 2006
# 657/D19 2006 through present

###1996-2005
#flow
#monthly.flow.rio.pres <-ts(monthly.flow.rio.pres, start=c(1975,1), frequency = 12)
#monthly.flow.rio.1996 <- window(monthly.flow.rio.pres, start=c(1996,1), end=c(2005,12))
#monthly.flow.west.pres <-ts(monthly.flow.west.pres, start=c(1975,1), frequency = 12)
#monthly.flow.west.1996 <- window(monthly.flow.west.pres, start=c(1996,1), end=c(2005,12))

flds=['NH4 mg/L N',
      'NO3 mg/L N',
      'PO4 mg/L P']

def lab_emp(station): # return EMP lab data for given station
    # takes care of dropping the station from the index
    l=lab2.loc[ (slice(None),station),: ]
    l.index=l.index.droplevel(1)
    return l

def lab_emp_mon(station): # return EMP lab data for given station, averaged to months
    return lab_emp(station).resample('M',how='mean')

sac_cols={}
sj_cols={}

d4=lab_emp_mon('D4')
c3=lab_emp_mon('C3')
d26=lab_emp_mon('D26')
d28a=lab_emp_mon('D28A')

for fld in flds:
    # D3,C3 have no NO3 data?
    # there are only 96 non-null nitrate samples in the EMP lab data,
    # and they are all from the 1970s.

    if fld=='NO3 mg/L N':
        # very few measurements of NO3 specifically, so approximate with
        # NO2+NO3:
        empfld='NOx mg/L N'
    else:
        empfld=fld

    if fld=='NH4 mg/L N':
        # Feb2016 update from EN slightly changes coefficients here
        sac=0.023448 + (0.162648*c3[fld])+(0.554124*d4[fld])

        # SJ uses linear regression over D26, D28A and D4
        # Updated Feb2016 regression:
        sj= -0.002152 + (0.319718*d26[fld]) + (0.234150*d28a[fld]) + (0.316507*d4[fld])
    elif fld=='NO3 mg/L N':
        # very few measurements of NO3 specifically, so approximate with
        # NO2/NO3:
        efld='NOx mg/L N'

        # aka D24 sub
        # Updated Feb2016 code - minor change in sig figs
        sac=-0.022851 + (0.199502*c3[empfld])+(0.808924*d4[empfld])

        # Likewise, minor differences at the 4th sigfig level.
        sj =(0.530542*d26[empfld]) + (0.161130*d28a[empfld]) + (0.381438*d4[empfld]) - 0.020406
    elif fld=='PO4 mg/L P':
        # Updated to this regression from Feb2016 code
        sac=0.010181 + (0.353006*c3[fld])+(0.451622*d4[fld])

        # Updated Feb2016
        sj= -0.002132 + (0.147484*d28a[fld]) + (0.311049*d4[fld]) + (0.544571*d26[fld])

    break1=pd.Timestamp("1995-12-01")
    # USGS is missing nitrogen data at the beginning of 2006, so stick
    # with C3/D4 an extra 9 months compared to Suisun loads.r
    break2=pd.Timestamp("2006-09-01")

    sel= ( (sac.index>break1) & (sac.index<break2) )
    sac_mid=sac[sel]
    sel= ( (sj.index>break1) & (sj.index<break2))
    sj_mid =sj[sel]

    # between 2006-01 and 2011-12, 
    # Sac gets concentration from usgs station 657, which appears to be reported in uM
    # SJ  gets concentration from EMP D19
    polaris=s.loc[ s.StationNumber==657, : ].groupby('Date').mean()

    if fld=='NH4 mg/L N':
        sac_later=polaris.nh* 14*(1e-3)
    elif fld=='NO3 mg/L N':
        # file doesn't have NO3, 
        sac_later=polaris.nn * 14*(1e-3)
    elif fld=='PO4 mg/L P':
        sac_later=polaris.p * 31*(1e-3) # fixed relative to 14 in Suisun loads.r
    sac_later=sac_later[ sac_later.index>=break2 ]
    
    sj_later=lab_emp_mon('D19')[empfld]
    sj_later=sj_later[sj_later.index>=break2]
                         
    sac_cols[fld]=pd.concat([sac_mid,sac_later])
    sj_cols[fld] =pd.concat([sj_mid,sj_later])

sac=pd.DataFrame(sac_cols)
sj =pd.DataFrame(sj_cols)

# union with flow data on daily timestep
# shouldn't need 'outer', but just in case.
sac_compiled=df_sac.join(sac,how='outer')
sj_compiled=df_sj.join(sj,how='outer')


# In[19]:


for df,name in [ (sac_compiled,'false_sac'),
                 (sj_compiled,'false_sj') ]:
    df=df.loc[ df.index> break1, : ].copy()
    assert isinstance(df.index,pd.DatetimeIndex)
    df.index.name='Date'
    df.rename(columns={'NH4 mg/L N':'NH3 mg/L N'},inplace=True)
    fn=os.path.join(compile_path,"%s.csv"%name)
    print("Writing %s"%fn)
    df.to_csv(fn)

