"""
Basic data quality check on the compiled data - 
 1. does each station have the data we expect?
 2. do the units appear to be correct?

"""
import os
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lp_filter
import glob
from collections import OrderedDict as odict

## 

def add_summer_noflow(df,gap_days=45,day_start=100,day_end=305):
    """ Designed for Napa, but possibly extend to others.
    Gaps of more than gap_days, which fall within the period
    dayofyear between [day_start,day_end] are filled with zero 
    flow.
    A new Dataframe is returned.
    """
    dt=np.diff(df.index)
    jumps=np.nonzero(dt>np.timedelta64(gap_days,'D'))[0]

    zero_flow_dates=[]
    for ji,jump in enumerate(jumps):
        a=df.index[jump]
        b=df.index[jump+1]
        print "%s - %s"%(a,b)

        # this is not so robust.
        if a.year!=b.year:
            print "Gap spans years"
            continue
        if a.dayofyear>100 and b.dayofyear<320:
            zero_flow_dates.append( pd.date_range(a,b,freq='D')[1:-1] )
        else:
            print "Summer gap: doesn't look like summer"

    zero_flow_dates=np.concatenate(zero_flow_dates)
    zero_df=pd.DataFrame( {'Date':zero_flow_dates} ).set_index('Date')

    zero_df['flow mgd']=0

    df_new=pd.concat( [df,zero_df] )
    df_new=df_new.sort()
    return df_new

## 
csvs=glob.glob('compiled_inputs/*.csv')
csvs=[s for s in csvs if 'stormwater' not in s]

dfs=odict()
for fn in csvs:
    name=os.path.basename(fn).replace('.csv','')
    dfs[name]=pd.read_csv(fn,parse_dates=['Date'],index_col='Date') 

##   Flows:

# Napa: generally no flow in summer, but in the dataframe this
# is just missing data, not 0 flow.
old_napa=dfs['napa']
dfs['napa']=add_summer_noflow(dfs['napa'])

if 0:
    plt.clf()
    plt.plot_date(old_napa.index,old_napa['flow mgd'],'g-o')
    plt.plot_date(dfs['napa'].index,dfs['napa']['flow mgd'],'b-o')

## 



# Notes for synthesizing complete timeseries:
# san_mateo: missing NH3 before 2008, has seasonal cycle afterwards.
# south_sf: great weekly pattern in NH3.
# pinole: missing data around 2009, or change in operations?
#         oddly sparse NO3 data.
# palo_alto: a lot of 0.2 - is that a MDL value?
# napa: loading study notes say that napa began denitrification in
#       2010 - so any extrapolation should not cross that time.

# ddsd: no3 data is just about useless.  probably better to get
#  very rough NH3:NO3 ratio, get seasonal from NH3, and go from there.
#  also there's a descreasing trend in discharge, 

## 


def seasonal_harmonic(ser,n_harmonics=3):
    """
    given a series with a datetime index, calculate
    a monthly climatology with the given number of
    harmonics.
    returns a function f: pd.Timestamp -> predicted value
    note that the first harmonic is the DC component
    """
    import harm_decomp

    omegas=np.arange(n_harmonics) * 2*np.pi/365.25

    dns=utils.dt64_to_dnum(col_df.index.values)
    omegas=np.arange(3) * 2*np.pi/365.25
    comps=harm_decomp.decompose(dns,col_df[col].values,omegas)
    # use default arguments to avoid late-binding issues
    def fit(times,comps=comps,omegas=omegas):
        if isinstance(times,pd.Index):
            times=times.values
        dns=utils.dt64_to_dnum(times)
        return harm_decomp.recompose(dns,comps,omegas)
    return fit

def monthly_climatology(ser):
    dfm=ser.groupby( ser.index.month ).mean()
    assert np.all(np.isfinite(ser)) # could come back and fill in with circ. interp.
    def fit(times,dfm=dfm):
        if not isinstance(times,pd.Index):
            times=pd.DatetimeIndex(times)
        month=times.month

        return dfm.ix[month].values
    return fit
        
## 
# Gap-filling for pinole flow:


df=dfs['pinole'] # sample

FLAG_ORIGINAL=1 # original dataset
FLAG_MONTHLY=2 # monthly climatology
FLAG_INTERP=4

# output synthesis is daily for this entire period.
# overkill, but there are some interesting features which
# wouldn't be resolved at monthly timescales
start_date=pd.Timestamp('2000-01-01')
end_date  =pd.Timestamp('2015-06-01')

#def synth(df):

expand=pd.date_range(start_date,end_date,freq='D')
syndf=pd.DataFrame({'Date':expand}).set_index('Date')


# Field specific portion:
col='flow mgd'
flag=col+" flag"
# narrow to valid values for this field
col_df=pd.DataFrame( {col:df[ np.isfinite(df[col]) ][col] } )
col_df[flag]=FLAG_ORIGINAL

# Add in climatology
f_fit=seasonal_harmonic(col_df[col])
# f_fit=monthly_climatology(col_df[col])

# Choose gaps large enough to revert to climatology
gap_days=20
m=syndf.loc[:,()].join( col_df )
m.fillna(method='ffill',limit=gap_days,inplace=True)
m.fillna(method='bfill',limit=gap_days,inplace=True)
climate_dates=m.index[ pd.isnull(m[col]) ]

climate_col=pd.DataFrame( {'Date':climate_dates,
                           col:f_fit(climate_dates),
                           flag:FLAG_MONTHLY} ).set_index('Date')



# Add in an adjustment for trends across gaps:
if 1:
    adjustment=col_df[col].values - f_fit(col_df.index)
    adjustment=utils.moving_average_nearest(utils.dt64_to_dnum(col_df.index.values),
                                            adjustment,
                                            n=3)
    adj_to_fit=np.interp( utils.dt64_to_dnum(climate_dates.values),
                          utils.dt64_to_dnum(col_df.index.values),adjustment)
    adj_expand=np.interp( utils.dt64_to_dnum(syndf.index.values),
                          utils.dt64_to_dnum(col_df.index.values),adjustment)
    climate_col.loc[:,col]+=adj_to_fit


df_new=pd.concat( [col_df,climate_col] )
df_new=df_new.sort()

syndf=syndf.join( df_new )
to_fill=np.isnan(syndf[col])
syndf.loc[ np.isnan(syndf[col]),flag]=FLAG_INTERP
# pandas 0.16 only fills in forward
syndf[col].interpolate(inplace=True,method='linear')
# so come back and copy backwards in time.
syndf.fillna(method='backfill',inplace=True)


plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
ax.plot_date(syndf.index,syndf[col],"k-",label='output')
sel=syndf[flag]==FLAG_ORIGINAL
scat=ax.plot(syndf.index[sel],syndf.loc[sel,col],'ko',label='observations')

ax.plot(syndf.index,
        f_fit(syndf.index) + adj_expand,
        'g-', label='trended climatology')

ax.legend()
