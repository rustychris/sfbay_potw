"""
Compile POTW Inputs

Should be run from scripts folder.

For each POTW, read in source data (`sources`, largely Loading Study data, plus 
HDR report for BACWA), apply any 
known corrections or unit conversions, and write out a cleaned CSV to
`outputs/intermediate`.

Based on Emily Novick's Loading Study scripts, *Suisun loads.r*, *SanPablo loads.r*, 
*CentralBay loads.r*, *LowerSouthBay loads.r*.

[converted from compile_bay_potw.ipynb]
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os
import six

# configure paths
sources_path="../sources"
compile_path='../outputs/intermediate'



# discard data before this time
start_date=pd.Timestamp('2000-01-01')

def clean_columns(df):
    renames={ c:c.strip()
              for c in df.columns }
    df.rename(columns=renames,inplace=True)
    return df

srcs={}

def small_plant(flow,NH3,NO3,PO4,name=None):
    """
    create a dataframe which has constant values for the 
    main constituents and flow.  Used in defining small plants
    where these values are estimated on design flow or similar.

    if name is given, the dataframe is also added to srcs[].

    if any of the specified values are actually a 12-element array,
    that's interpreted as a seasonal pattern and repeated annually.
    """
    # oddly, first entry is end of month...
    # this had stopped at 2012-01-01 - why not go further?
    dates=pd.date_range(start=start_date,end='2017-01-01',freq='M')
    def expand(v):
        v=np.atleast_1d(v)
        if len(v)==1:
            v=v*np.ones(len(dates))
        elif len(v)==12:
            v=v[ dates.month-1 ]
        else:
            raise Exception("only scalar or monthly climatology allowed")
        return v
            
    df=pd.DataFrame({'Date':dates,
                     'NH3 mg/L N':expand(NH3),
                     'NO3 mg/L N':expand(NO3),
                     'PO4 mg/L P':expand(PO4),
                     'flow mgd':expand(flow)})
    df=df.set_index('Date')

    if name is not None:
        srcs[name]=df

    return df



## San Jose

df = pd.read_csv(os.path.join(sources_path,'loading_study','SanJose.csv'),
                 parse_dates=['Date'])
clean_columns(df)

df.PO4 *= (31./95) #convert historic data from mg/L PO4 to mg/L P
df.OrthoP *= (31./95)

# EN code removes 2006 PO4 values from consideration (change in treatment)
# but here keep them for model forcing

# estimates - no mention of source in LowerSouthBay loads.r, or the Loading
# Study.
df['TKN mg/L N']=1.78
df['TP mg/L P']=0.57

# make all the units explicit
df.rename(columns={'Flow':'flow mgd',
                   'TSS':'TSS mg/L',
                   'NH3' :'NH3 mg/L N',
                   'NO3' :'NO3 mg/L N',
                   'NO2' :'NO2 mg/L N',
                   'PO4' :'PO4 mg/L P'},
          inplace=True)

df=df.set_index('Date')

srcs['san_jose']=df


## Palo Alto

df = pd.read_csv(os.path.join(sources_path,'loading_study','PaloAlto.csv'),
                 parse_dates=['Date'])
clean_columns(df)

# convert historic PO4 from mg/L PO4 to mg/L P
df.PO4 *= (31./95)

# estimates
df['TKN mg/L N'] = 0.87
df['TP mg/L P'] = 4.31

# make all the units explicit
df.rename(columns={'Flow':'flow mgd',
                   'NH3' :'NH3 mg/L N',
                   'NO3' :'NO3 mg/L N',
                   'NO2' :'NO2 mg/L N',
                   'PO4' :'PO4 mg/L P'
               },
          inplace=True)

df=df.set_index('Date')

srcs['palo_alto']=df


## Sunnyvale

df = pd.read_csv(os.path.join(sources_path,'loading_study','Sunnyvale.csv'),
                 parse_dates=['Date'])
clean_columns(df)

# estimates - 2012 PO4 not representative, so scaled historic TP by % PO4
df['PO4'] = df.TP*0.93

# make all the units explicit
df.rename(columns={'Flow':'flow mgd',
                   'NH3' :'NH3 mg/L N',
                   'NO3' :'NO3 mg/L N',
                   'NO2' :'NO2 mg/L N',
                   'PO4' :'PO4 mg/L P',
                   'TKN' :'TKN mg/L N',
                   'OrgN':'OrgN mg/L N',
                   'TP'  :'TP mg/L P',
                   'Temp':'temperature degC',
                   'TSS':'TSS mg/L',
                   'TDS':'TDS mg/L'
               },
          inplace=True)

df=df.set_index('Date')

srcs['sunnyvale']=df


## Stormwater

# Currently not handled in this dataset - instead, rough loading estimates are 
# attached to surface water flows outside the POTW framework.

# Stormwater - disabled.

# df=pd.read_csv(os.path.join( loadstudy_dir,'..','Results ModelRun_02282013.csv'))

## SF Southeast

df=pd.read_csv(os.path.join(sources_path,'loading_study','SFSoutheast.csv'),
               parse_dates=['Sample_Date'])
df.rename(columns={'Sample_Date':'Date'},inplace=True)

# NOTES: for historical reporting - 1996 data no good - flowmeter malfunctioning 
# (Amy Chastain, pers comm)

cols=[]

for v in df.Parameter.unique():
    col=df[df.Parameter==v].set_index('Date')['Result'].to_frame(v)
    cols.append(col)

df2=cols[0].join(cols[1:],how='outer')
df,orig=df2,df

# Original code calculated loads based on flows only on days that the corresponding
# nutrient was measured.

df.rename(columns={'Flow':'flow mgd',
                   'TKN':'TKN mg/L N',
                   'NO3-N':'NO3 mg/L N',
                   'NO2-N':'NO2 mg/L N',
                   'NH3-N':'NH3 mg/L N',
                   'PHOSPHORUS_T':'TP mg/L P',
                   'PO4_ORTHO-P':'PO4 mg/L P',
                   'PH':'pH',
                   'TSS':'TSS mg/L'},
          inplace=True)

srcs['sf_southeast']=df


## EBDA

df=pd.read_csv(os.path.join(sources_path,'loading_study','EBDA.csv'),
               parse_dates=['Date'])

cols=[]

for v in df.Pollutant.unique():
    col=df[df.Pollutant==v].set_index('Date')['Value'].to_frame(v)
    cols.append(col)

df2=cols[0].join(cols[1:],how='outer')

df,orig=df2,df

# estimates
df['NO3 mg/L N']=2.34
df['TKN mg/L N']=31.00
df['PO4 mg/L P']=1.67
df['TP mg/L P']=2.39

df.rename(columns={'Ammonia, Total (as N)':'NH3 mg/L N',
                   'Flow':'flow mgd'},
          inplace=True)

srcs['ebda']=df


## San Mateo

df=pd.read_csv(os.path.join(sources_path,'loading_study','SanMateo.csv'),
               parse_dates=['Date'])


# seems that this covers all the records, but just to be safe, I suppose
sel=( (df.Description=='E-001 (Dry) Eff Monthly Average') |
      (df.Description=='E-001 (Wet) Eff Monthly Average') )
df=df[sel]

cols=[]

for v in df.Pollutant.unique():
    col=df[df.Pollutant==v].set_index('Date')['Value'].to_frame(v)
    cols.append(col)

df2=cols[0].join(cols[1:],how='outer')

df,orig=df2,df

# estimates from R code
df['NO3 mg/L N']=1.64
df['TKN mg/L N']=32.58
df['PO4 mg/L P']=2.63
df['TP mg/L P']=3.02

df.rename(columns={'NH3':'NH3 mg/L N',
                   'Flow':'flow mgd'},
          inplace=True)

srcs['san_mateo']=df


## South Bayside

df=pd.read_csv(os.path.join(sources_path,'loading_study','SouthBayside.csv'),
               parse_dates=['Date'])

# estimates
df['NO3 mg/L N']= 0.56
df['TKN']= 40.75 # original file has TKN, but only 14 data points
df['PO4 mg/L P']= 3.99
df['TP mg/L P'] = 3.49

df.rename(columns={'TKN':'TKN mg/L N',
                   'Temp':'temperature degC',
                   'NH3':'NH3 mg/L N',
                   'TSS':'TSS mg/L',
                   'Flow':'flow mgd'},
          inplace=True)

srcs['south_bayside']=df


## South SF

df=pd.read_csv(os.path.join(sources_path,'loading_study','SouthSF.csv'),
               parse_dates=['Date'])


# estimates
df['NO3 mg/L N'] = 1.91
df['TKN mg/L N'] = 32.0
df['PO4 mg/L P'] = 3.0
df['TP mg/L P']  = 4.11

df.rename(columns={'Temp':'temperature degC',
                   'NH3':'NH3 mg/L N',
                   'TSS':'TSS mg/L',
                   'Flow':'flow mgd'},
          inplace=True)

srcs['south_sf']=df


## Small Plants


# burlingame:
df=small_plant(flow=3.67,
               NH3=22.70,
               NO3=4.52,
               PO4=2.48,
               name='burlingame')

# millbrae
df=small_plant(flow=2.0,
               NH3=39.17,
               NO3=0.1,
               PO4=2.60,
               name='millbrae')

# sfo
df=small_plant(flow=1.5,
               NH3=40.46,
               NO3=3.76,
               PO4=2.35,
               name='sfo')

# # livermore - commented out in R code
# df=small_plant(flow=5.7,
#                NH3=44.0,
#                NO3=0.02,
#                PO4=0.81,
#                name='livermore')
# 


## Central Bay
 
## EBMUD

df=pd.read_csv(os.path.join(sources_path,'loading_study','EBMUD.csv'),
               parse_dates=['CDATE'])
df.rename(columns={'CDATE':'Date'},inplace=True)


# because of the variation in SITE, SAMTYPE, LOCATOR, can't 
# use the same pivot method as above.

cols=[]

sel_flow= (df.PARM_STORED=='Flow') & (df.SITE=='E-001  Eff Daily Average')
col=df[sel_flow].set_index('Date')['NUMVALUE'].to_frame('flow mgd')
cols.append(col)

sel= ( (df.PARM_STORED=='AMMONIA AS N') &
       df.SAMTYPE.isin(['CF01','CF03','CFV','CTV','COMP']) &
       df.LOCATOR.isin(['EFF EPS 04', 'EFF DECHLOR']) )

col=df[sel].set_index('Date')['NUMVALUE'].to_frame('NH3 mg/L N')
cols.append(col)

# R code didn't include TSS, but it's there in the same set of samples as
# the NH3 numbers are using
sel= ( (df.PARM_STORED=='TOTAL SUSPENDED SOLIDS') &
       df.SAMTYPE.isin(['CF01','CF03','CFV','CTV','COMP']) &
       df.LOCATOR.isin(['EFF EPS 04', 'EFF DECHLOR']) )

col=df[sel].set_index('Date')['NUMVALUE'].to_frame('TSS mg/L')
cols.append(col)

df2=cols[0].join(cols[1:],how='outer')
 
df,orig=df2,df

# estimates
df['NO3 mg/L N']=4.46
df['TKN mg/L N']=40.29
df['PO4 mg/L P']=2.91
df['TP mg/L P']=4.51

# NOTE: code fragment in R code about varying PO4 loads with high/low flow.
srcs['ebmud']=df


## West County / Richmond

df=pd.read_csv(os.path.join(sources_path,'loading_study','WestCounty_Richmond.csv'),
               parse_dates=['Date'])

cols=[]

col=df[(df.Pollutant=='Flow')&(df.Description=='E-001 DC-Combined) Eff Daily Average')]
col=col.set_index('Date')['Value'].to_frame('flow mgd')
cols.append(col)

col=df[(df.Pollutant=='NH3')&(df.Description=='E-001 DC-Combined) Eff Daily Maximum')]
col=col.set_index('Date')['Value'].to_frame('NH3 mg/L N')
cols.append(col)

df2=cols[0].join(cols[1:],how='outer')

df,orig=df2,df

# estimates
df['NO3 mg/L N']=3.19
df['TKN mg/L N']=22.85
df['PO4 mg/L P']=1.60
df['TP mg/L P']=1.81

srcs['west_county_richmond']=df


## Central Marin

df=pd.read_csv(os.path.join(sources_path,'loading_study','CentralMarin.csv'),
               parse_dates=['Date'])

# estimates
df['NO3 mg/L N'] = 2.92
df['TKN mg/L N'] = 33.36
df['PO4 mg/L P'] = 3.34
df['TP mg/L P']  = 3.61

df.rename(columns={'Flow':'flow mgd',
                   'NH3':'NH3 mg/L N',
                   'TSS':'TSS mg/L',
                   'Temp':'temperature degC'},
          inplace=True)

srcs['central_marin']=df


## Smaller Plants

# sasm
df=small_plant(flow=2.4,
               NH3=3.7,
               NO3=15.8,
               PO4=4.4,
               name='sasm')

# sausalito
df=small_plant(flow=1.2,
               NH3=8.76,
               NO3=12.23,
               PO4=3.57,
               name='sausalito')

# treasure island
df=small_plant(flow=1.3,
               NH3=0.45,
               NO3=7.41,
               PO4=3.07,
               name='treasure_island')

# marin 5
df=small_plant(flow=0.65,
               NH3=21.0,
               NO3=0.5,
               PO4=2.54,
               name='marin5')


## San Pablo Bay

## Napa

df=pd.read_csv(os.path.join(sources_path,'loading_study','Napa.csv'),
               parse_dates=['Date'])

# NOTE: pretty sure they only discharge for part of the year
# estimates
df['NO3 mg/L N'] = 6.26
df['TKN mg/L N'] = 4.52
df['PO4 mg/L P'] = 1.06
df['TP mg/L P']  = 1.33

df.rename(columns={'Flow':'flow mgd',
                   'NH3':'NH3 mg/L N'},
          inplace=True)
srcs['napa']=df


## Vallejo

df=pd.read_csv(os.path.join(sources_path,'loading_study','Vallejo.csv'),
               parse_dates=['Date'],
               na_values=[' ']) # pH has some spaces.

# estimate
df['PO4 mg/L P'] = 2.69

df['TSS']=df.TSS.astype(np.float64)

df.rename(columns={'Flow':'flow mgd',
                   'NH3':'NH3 mg/L N',
                   'NO2':'NO2 mg/L N',
                   'NO3':'NO3 mg/L N',
                   'TKN':'TKN mg/L N',
                   'TP' :'TP mg/L P',
                   'Temp':'temperature degC',
                   'TSS':'TSS mg/L'},
          inplace=True)

srcs['vallejo']=df

## Chevron

# NOTE: 'ND' should probably be some nominal zero-ish value.
df=pd.read_csv(os.path.join(sources_path,'loading_study','Chevron.csv'),
               parse_dates=['Day'],
               na_values=['ND'])
df.rename(columns={'Day':'Date'},inplace=True)

# estimates
df['NO3 mg/L N'] =14.67
df['TKN mg/L N'] = 3.05
df['PO4 mg/L P'] = 1.83
df['TP mg/L P']  = 2.83

df.rename(columns={'Flow':'flow mgd',
                   'NH3':'NH3 mg/L N'},
          inplace=True)

srcs['chevron']=df


## Shell

df=pd.read_csv(os.path.join(sources_path,'loading_study','Shell.csv'),
               parse_dates=['Date'],
               na_values=['ND'])

cols=[]

col=df[(df.Pollutant=='Flow')&(df.Description=='E-001 Eff Daily Average')]
col=col.set_index('Date')['Value'].to_frame('flow mgd')
cols.append(col)

col=df[df.Pollutant=='Ammonia'].set_index('Date')['Value'].to_frame('NH3 mg/L N')
cols.append(col)

col=df[df.Pollutant=='Nitrate'].set_index('Date')['Value'].to_frame('NO3 mg/L N')
cols.append(col)

df2=cols[0].join(cols[1:],how='outer')

df,orig=df2,df

# estimates
df['TKN mg/L N']=3.95
df['PO4 mg/L P']=0.01
df['TP mg/L P']=0.15

srcs['shell']=df


## Novato

df=pd.read_csv(os.path.join(sources_path,'loading_study','Novato.csv'),
               parse_dates=['Date'])

# estimates
df['NO3 mg/L N'] = 11.04
df['TKN mg/L N'] = 1.44
df['PO4 mg/L P'] = 0.37
df['TP mg/L P'] = 0.39

df.rename(columns={'Flow':'flow mgd',
                   'NH3':'NH3 mg/L N'},
          inplace=True)
                   
srcs['novato']=df


## Sonoma Valley

df=pd.read_csv(os.path.join(sources_path,'loading_study','SonomaValley.csv'),
               parse_dates=['DATE'],
               na_values=[' - - -'])
df.rename(columns={'DATE':'Date'},inplace=True)


# goofy 6.4/6.6 entries in pH:
def fix_slash(s):
    if isinstance(s,str):
        if "/" in s:
            s=np.mean([float(t) for t in s.split('/')])
        else:
            s=float(s)
    return s
df['pH']=df.pH.apply(fix_slash)
df.ix[ df.Temp=="19..1",'Temp' ] = "19.1"
df.ix[ df.Temp=="24.8.",'Temp' ] = "24.8"
df.ix[ df.Temp=="21. 8",'Temp' ] = "21.8"
df.ix[ df.Temp==' ','Temp' ] = "nan"
df['Temp']=df.Temp.astype(np.float64)

# estimates
df['PO4 mg/L P'] = 2.60 # file has some PO4 data, but following R code's lead.

df.rename(columns={'FLOW ':'flow mgd',
                   'TSS':'TSS mg/L',
                   'NO3':'NO3 mg/L N',
                   'OrgN':'orgN mg/L N',
                   'TKN':'TKN mg/L N',
                   'PO4':'PO4 mg/L',
                   'TP':'TP mg/L P',
                   'Temp':'temperature degC',
                   'NH3':'NH3 mg/L N'},
          inplace=True)
                   
srcs['sonoma_valley']=df


## Phillips 66

df=pd.read_csv(os.path.join(sources_path,'loading_study','Phillips66.csv'),
               parse_dates=['Date'])

# estimates
df['TKN mg/L N'] = 0.89
df['PO4 mg/L P'] = 0.43
df['TP mg/L P']  = 0.56

df['pH']=0.5*(df.minpH + df.maxpH)
# data look like Fahrenheit
temp_degF=0.5*(df.minTemp + df.maxTemp)
df['temperature degC']=(temp_degF-32)*5./9

df.rename(columns={'Flow':'flow mgd',
                   'NH3':'NH3 mg/L N',
                   'NO3':'NO3 mg/L N',
                   'TSS':'TSS mg/L'},
          inplace=True)
                   
srcs['phillips66']=df


## Pinole  

df=pd.read_csv(os.path.join(sources_path,'loading_study','Pinole.csv'),
               parse_dates=['Date'])

df.rename(columns={'Flow':'flow mgd',
                   'TKN':'TKN mg/L N',
                   'NH3':'NH3 mg/L N',
                   'NO3':'NO3 mg/L N',
                   'NO2':'NO2 mg/L N',
                   'Temp':'temperature degC',
                   'TSS':'TSS mg/L'},
          inplace=True)

# estimates
df['TKN mg/L N'] =25.50 
df['PO4 mg/L P'] = 3.22
df['TP mg/L P']  = 3.35

srcs['pinole']=df


## Smaller plants

# no summer discharge
small_plant(flow=[2.67]*4 + [0]*6 + [2.67]*2,
            NH3=0.27,NO3=9.63,PO4=3.80,
            name='american')
               
small_plant(flow=3.0,NH3=25.0,NO3=0.93,PO4=2.80,name='benicia')

small_plant(flow=[0.56]*5 + [0]*5 + [0.56]*2,
            NH3=2.84,NO3=10.97,PO4=2.20,
            name='calistoga')

small_plant(flow=0.96,NH3=4.68,NO3=2.87,PO4=1.43,name='ch')

small_plant(flow=[1.95]*5 + [0]*5 + [1.95]*2,
             NH3=2.73,NO3=20.33,PO4=3.60,
             name='lg')

small_plant(flow=[3.5]*4 + [0]*6 + [3.5]*2,
            NH3=0.45,NO3=1.02,PO4=2.38,
            name='petaluma')

small_plant(flow=0.7,NH3=2.47,NO3=11.23,PO4=3.67,name='rodeo')
small_plant(flow=[0.33]*4 + [0]*6 + [0.33]*2,
            NH3=8.3,NO3=0.1,PO4=3.10,
            name='st_helena')
small_plant(flow=1.95,NH3=0.34,NO3=20.83,PO4=0.02,name='valero')

small_plant(flow=[0.37]*5+[0]*5+[0.37]*2,
            NH3=6.25,NO3=13.00,PO4=3.00,
            name='yountville') ;


##  Suisun Bay - CCCSD

# the results of this are already added into NH4 mass balance csv
# (I think that comment is left over from the R source, and not
# relevant here)
df=pd.read_csv(os.path.join(sources_path,'loading_study','CCCSD.csv'),
               parse_dates=['Date'],
               na_values=['N.M.','rejected','not analyzed','<0.10','.','<0.50'])

# estimate
df['PO4 mg/L P']=0.52

df.rename(columns={'Flow':'flow mgd',
                   'Temp':'temperature degC',
                   'TP':'TP mg/L P',
                   'TSS':'TSS mg/L',
                   'NH3':'NH3 mg/L N',
                   'TKN':'TKN mg/L N',
                   'NO2':'NO2 mg/L N',
                   'NO3':'NO3 mg/L N'},
          inplace=True)

srcs['cccsd']=df


##  Fairfield Suisun

df=pd.read_csv(os.path.join(sources_path,'loading_study','FS.csv'),
               parse_dates=['Date'],na_values=['ND'])

df.rename(columns={'NH3':'NH3 mg/L N',
                   ' TKN':'TKN mg/L N',
                   'NO3':'NO3 mg/L N',
                   'NO2':'NO2 mg/L N',
                   'TN':'TN mg/L N',
                   'TP':'TP mg/L P',
                   '  Flow':'flow mgd'},
          inplace=True)
                   
# estimate
df['PO4 mg/L P']=3.86

srcs['fs']=df


## DDSD: Delta Diablo

df=pd.read_csv(os.path.join(sources_path,'loading_study','DDSD.csv'),
               parse_dates=['Date'])

df.rename(columns={'Flow':'flow mgd',
                   'NH3':'NH3 mg/L N',
                   'NO3':'NO3 mg/L N',
                   'NO2':'NO2 mg/L N',
                   'TP':'TP mg/L P'},
          inplace=True)

# estimates
df['TKN mg/L N'] =27.85
df['PO4 mg/L P'] = 0.79

srcs['ddsd']=df


## Tesoro

df=pd.read_csv(os.path.join(sources_path,'loading_study','Tesoro.csv'),
               parse_dates=['Date'])

cols=[]

col=df[(df.Pollutant=='Flow')&(df.Description=='E-001 Eff Daily Average')]
col=col.set_index('Date')['Value'].to_frame('flow mgd')
cols.append(col)

col=df[(df.Pollutant=='NH3')&(df.Description=='E-001 Eff Daily Maximum')&(df.Unit=='mg/l')]
col=col.set_index('Date')['Value'].to_frame('NH3 mg/L N')
cols.append(col)

df2=cols[0].join(cols[1:],how='outer')
df,orig=df2,df

#estimations
df['NO3 mg/L N']=0.82
df['TKN mg/L N']=6.75
df['PO4 mg/L P']=0.06
df['TP mg/L P']=0.24

srcs['tesoro']=df


## Smaller plants

small_plant(flow=2.1,NH3=0.58,NO3=21.29,PO4=3.61,
            name='mt_view') ; 


## 


# Light housecleaning:
for name in srcs:
    df=srcs[name]
    if df.index.name != 'Date':
        df=df.set_index('Date')

    sel=(df.index.values > start_date.to_datetime64())
    df=df[sel]

    srcs[name]=df

## 

# Recent Data 
# *(but not yet the HDR data)*

# Now bring in the more recent data
eff_df=pd.read_csv(os.path.join(sources_path,
                                "final effluent_concentrations_Mar2015.csv"),
                   parse_dates=['Date'],
                   na_values=['#DIV/0!'])

def round_to_day(t):
    return pd.Timestamp(t.date())

# round date to day so the joins work better.
eff_df['Date'] = eff_df.Date.apply(round_to_day)

# final effluent_concentrations_Mar2015.csv
# is missing temperature.
# but 
# final effluent_concentrations_Sep2014.csv
# has Temp min, Temp max.

# hmm - pH has some values greater than 14. Those all belong to Central Marin.
# Temperature means suggest that Rodeo reports Fahrenheit, and Central Marin
# min temp is no good.
eff_2014=pd.read_csv(os.path.join(sources_path,
                                  "final effluent_concentrations_Sep2014.csv"),
                     parse_dates=['Date'],
                     na_values=['#DIV/0!','Jan-00'])

# Some of the dates have times, most don't.  All appear to start with
# M/D/YYYY

dates=[pd.Timestamp(s.split()[0].replace('5014','2014'))
       for s in eff_2014['Date'].values]
eff_2014['Date']=dates

cm=(eff_2014.Facility=='Central Marin Sanitation Agency')
eff_2014.loc[ cm,['Temp min','Temp max','pH Max','pH min'] ]=np.nan

rod=(eff_2014.Facility=='Rodeo Sanitary District')
for fld in ['Temp min','Temp max']:
    eff_2014.loc[ rod,fld] = (eff_2014.loc[ rod,fld]-32)*5./9


def first_valid(*args):
    vals=args[0].copy()
    for a in args[1:]:
        missing=np.isnan(vals)
        vals[missing]=a[missing]
    return vals

eff_2014['temperature degC'] = first_valid( 0.5*(eff_2014['Temp max'] + eff_2014['Temp min']),
                                            eff_2014['Temp max'],
                                            eff_2014['Temp min'] )
eff_2014['pH'] = first_valid(0.5*(eff_2014['pH Max'] + eff_2014['pH min']),
                             eff_2014['pH Max'],
                             eff_2014['pH min'])

# limit to the columns we care about:
eff_update=eff_2014.loc[:,['Date','Facility','temperature degC','pH']]

# make a copy for idempotency
srcs1=dict(srcs)

def add_newer(name,facility,merge='error'):
    """
    merge: 'error': overlapping data causes an exception
    'old': use old data in overlapping regions
    'new': use new data in overlapping regions
    """
    df=srcs[name]
    df_newer=eff_df[ eff_df.Facility==facility].groupby('Date').mean()
    df_update=eff_update[ eff_update.Facility==facility ].groupby('Date').mean()
    df_newer=df_newer.join(df_update,rsuffix='_2014',how='outer')

    # in case this is run multiple times, replaces recent records
    old_overlap_count=len( df.ix[ df.index>=df_newer.index.min() ] )
    new_overlap_count=len( df_newer.ix[ df_newer.index<=df.index.max() ] )

    if old_overlap_count:
        msg="%s: %d old and %d new records overlap"%(facility,
                                                     old_overlap_count,
                                                     new_overlap_count)
        if merge == 'error':
            raise Exception(msg)
        elif merge=='old':
            df_newer=df_newer.ix[ df_newer.index>df.index.max() ]
        elif merge=='new':
            df=df.ix[ df.index<df_newer.index.min() ]
        else:
            assert False
        print(msg)

    df=df[ df.index < df_newer.index.min() ]
    df_newer.rename(columns={ 'Flow  (MGD)':'flow mgd',
                              'Peak Flow (MGD)':'peak_flow mgd',
                              'TN (mg/L)':'TN mg/L N', 
                              'TDN (mg/L)':'TDN mg/L N',
                              'TKN (mg/L)':'TKN mg/L N',
                              'SKN (mg/L)':'SKN mg/L N',
                              'NO3 (mg/L)':'NO3 mg/L N', 
                              'NO2 (mg/L)':'NO2 mg/L N',
                              'Total NH3 (mg/L)':'NH3 mg/L N',
                              'Urea* (mg/L)':'urea* mg/L N',  # NOTE: assumption
                              'TP (mg/L)':'TP mg/L P', 
                              'TDP (mg/L)':'TDP mg/L P',
                              'DRP** (mg/L)':'DRP** mg/L P',
                              'TSS (mg/L)':'TSS mg/L',
                              'TRP (mg/L)':'TRP mg/L P'},
                    inplace=True)
    # in a few cases (DDSD, looking at you) there are notes in weird places,
    # like the TRP field.
    def float_or_bust(f):
        try:
            return float(f)
        except ValueError:
            return np.nan
    for fld in ['TRP mg/L P']:
        if fld in df_newer.columns and df_newer[fld].dtype == object:
            df_newer[fld]=df_newer[fld].apply(float_or_bust)
    df=pd.concat([df,df_newer])
    srcs1[name]=df
    return df


# Check on central_marin and sonoma_valley - the old data significantly
# overlaps with the new.  Central Marin: the new data is monthly, but has
# a lot more parameters.  No appreciable variation which is captured by the
# old daily data and not the new monthly, so let the old just get replaced
# in the case of Sonoma Valley, it's just one record of the new data, 
# and a lot of the old.  Keep the old data.
# San Jose: old data is monthly, new data is bi-weekly. or is it semi-monthly?
# Any plants specified above with small_plant() should have merge='new' here,
# since the old data is just a periodic estimate

add_newer('american','City of American Canyon',merge='new')
add_newer('benicia','City of Benicia WWTP',merge='new')
add_newer('calistoga','City of Calistoga WWTP',merge='new')
add_newer('cccsd','Central Contra Costa Sanitary District')
add_newer('chevron','Chevron Richmond Refinery')
add_newer('burlingame','City of Burlingame WWTF',merge='new')
add_newer('st_helena','City of St. Helena Waste Water Treatment Plant',merge='new')
add_newer('sunnyvale','City of Sunnyvale')
add_newer('palo_alto','City of Palo Alto RWQCP')
add_newer('petaluma','City of Petaluma',merge='new')
add_newer('central_marin','Central Marin Sanitation Agency',merge='new')
add_newer('ddsd','Delta Diablo Sanitation District')
add_newer('ebda','EBDA')
add_newer('ebmud','East Bay Municipal Utility District')
add_newer('fs','Fairfield - Suisun Sewer District')
add_newer('pinole','Pinole-Hercules WPCP')
add_newer('lg','Las Gallinas Valley Sanitary District',merge='new')
add_newer('millbrae','City of Millbrae WPCP',merge='new')
add_newer('mt_view','Mt. View Sanitary District',merge='new')
add_newer('novato','Novato Sanitary District')
add_newer('napa','Napa Sanitation District')
add_newer('phillips66','Phillips 66 San Francisco Refinery')
add_newer('rodeo','Rodeo Sanitary District',merge='new')
add_newer('san_mateo','City of San Mateo')
add_newer('sasm','Sewerage Agency of Southern Marin',merge='new')
add_newer('sausalito','Sausalito - Marin City Sanitary District',merge='new')
add_newer('sfo','San Francisco International Airport - MLTP',merge='new')
add_newer('shell','Shell Martinez Refinery')
add_newer('san_jose','San Jose/Santa Clara Water Pollution Control Plant',merge='new')
add_newer('sf_southeast','Southeast Water Pollution Control Plant CCSF')
add_newer('south_sf','South San Francisco-San Bruno Water Quality Control Plant')
add_newer('sonoma_valley','Sonoma Valley County Sanitation District',merge='old')
add_newer('treasure_island','Treasure Island Water Pollution Control Plant',merge='new')
add_newer('valero','Valero Refining Company - CA',merge='new')
add_newer('vallejo','Vallejo Sanitation & Flood Control District')
add_newer('west_county_richmond','West County Agency')
add_newer('yountville','Town of Yountville',merge='new')
add_newer('tesoro','Tesoro Golden Eagle Refinery')
add_newer('marin5','Sanitary District No.5 of Marin County Main Plant',merge='new')
# This plant is jointly operated by Crockett and C&H, but I believe owned
# by C&H.
add_newer('ch','Phillip F. Meads Water Treatment Plant',merge='new')
# changed names 2015/06
add_newer('south_bayside','Silicon Valley Clean Water')

# There is also Marin SD5 Paradise Cove, but it's flow is tiny.  Really tiny.
#add_newer('XXX','Sanitary District No.5 of Marin County Paradise Cove Plant')

# compared to the old csv_flows files, only sf-north_point.csv appears to be missing.

## 

# Write final csvs:
for name in srcs1:
    srcs1[name].to_csv(os.path.join(compile_path,'%s.csv'%name))


## HDR Data
# 
# These have already been processed to a degree by HDR.  Each value represents an average flow or
# load for the given month and discharge.  


# This file includes data from 2012 through mid-2016
xl_fn="../sources/BACWA_HardGARdata_20161027.xlsx"
raw_df=pd.read_excel(xl_fn,sheetname=0,header=None,parse_cols="E:AV",skiprows=15)
raw=raw_df.values


def is_any_str(x):
    for t in six.string_types:
        if isinstance(x,t):
            return True
    return False

block_starts=[0] + [i+1 
                    for i in range(raw.shape[0]-1)
                    if raw[i,3] is np.nan and is_any_str(raw[i+1,3])]
block_ends=block_starts[1:] + [raw.shape[0]]
block_starts=np.array(block_starts)

blocks=[]
for blk_start,blk_end in zip(block_starts,block_ends):

    hdr_df=raw_df.iloc[blk_start:blk_start+3]
    blk_df=raw_df.iloc[blk_start+3:blk_end] # 3 rows of header info

    hdr_df.head()
    blk_unit=hdr_df.iloc[0,1]
    blk_var=hdr_df.iloc[0,3]
    long_names=hdr_df.iloc[1,3:]
    short_names=hdr_df.iloc[2,3:]

    # mark columns to remove with nan label
    blk_df.columns=['month','year','nan']+list(short_names)

    valid=[str(f) != 'nan' for f in blk_df.columns]
    blk_df=blk_df.iloc[:,valid]
    blk_df['analyte']=blk_var+"_"+blk_unit
    blocks.append(blk_df)


# time and analyte are in rows, station is in columns
df=pd.concat(blocks)

# moves the site name columns into a long format dataframe,
# preserving analyte, year, month to identify rows.
df4=pd.melt(df,id_vars=['analyte','year','month'],var_name='site')
df5=df4[ ~df4.value.isnull() ]
df5.head()


renames={'Ammonia, Total (as N)_kg/day':'ammonia_kgN_per_day',
         'Flow_mgd':'flow_mgd',
         'TKN_kg/day':'TKN_kgN_per_day',
         'Nitrite Plus Nitrate (as N)_kg/day':'NOx_kgN_per_day',
         'Nitrogen, Total (as N)_kg/day':'total_kgN_per_day',
         'Phosphorus, Total (as P)_kg/day':'total_kgP_per_day',
         'Orthophosphate, Dissolved (as P)_kg/day':'diss_OrthoP_kgP_per_day'}
df6=df5.copy()
df6['analyte']=df6.analyte.map(renames)
df6['year'] = df6.year.astype('i4')

# make sure we didn't accidentally drop some
assert len(df6.analyte.unique()) == len(df5.analyte.unique())


#  A few minor data cleaning steps, per an email with Mike Falk, possible 
#  steps are:
#  
#   * Paradise Cove and Tiburon have some periods of missing flow
#     data. Use the average of flows from that month, but in other
#     years.  Since this is largely what will be done in synthesize,
#     don't worry about it here.
#
#   * For months missing a load but having a flow, calculate
#     concentration from adjacent months, combine with flow to get
#     load.
#
#   * Crockett has only two data points for load - better to calculate
#     a mean concentration, apply it for the rest of the time.

# These are just as well handled in the synthesize code as here.
## 

# # Bring in the updated 2016-17 HDR data
# xl_fn="../sources/BACWA_GAR_4SFEI.XLSM"
# raw_df=pd.read_excel(xl_fn,sheetname=4, # "Copied2016_2017Load"
#                      header=None,parse_cols="F:AW",skiprows=15)
# raw=raw_df.values

## 

# need to fix the datatypes of year,value
missing=df6.value=='--'
df6.loc[missing,'value']=np.nan
print("%d missing values: -- => nan"%np.sum(missing))
display(df6.loc[missing,:].groupby(['site','analyte']).sum())

nd=df6.value=='ND'
df6.loc[nd,'value']=0.0 # one point - don't worry about it
print("%d nondetect values: ND => 0"%np.sum(nd))
display(df6.loc[nd,:].groupby(['site','analyte']).sum())

for v in df6.value:
    try:
        float(v)
    except:
        print(v)
        
df6['value']=df6.value.astype('f4')


# In[ ]:


df6.to_csv(os.path.join(compile_path,'hdr_parsed_long.csv'),index=False)

