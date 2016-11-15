"""
Process all of the CSVs, writing out an updated/filled csv
"""
import os
import pdb
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lp_filter
import glob
from collections import OrderedDict as odict

## 
FLAG_ORIGINAL=1 # original dataset
FLAG_MONTHLY=2 # monthly climatology
FLAG_INTERP=4
FLAG_SEASONAL_ZERO=8 # assumed zero based on time of year
FLAG_NULL=-1

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

    flag='flow mgd flag'
    zero_df[flag]=0
    zero_df[flag]=FLAG_SEASONAL_ZERO
    if flag not in df.columns:
        # explicitly create it b/c otherwise it will be cast to float
        # in order to store nan.
        df[flag]=FLAG_NULL # i.e. integer version of nan - see below
    
    df_new=pd.concat( [df,zero_df] )
    df_new=df_new.sort_index()

    return df_new

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

    dns=utils.dt64_to_dnum(ser.index.values)
    omegas=np.arange(3) * 2*np.pi/365.25
    comps=harm_decomp.decompose(dns,ser.values,omegas)
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


def monthly_climatology_lp(ser):
    dfm=ser.groupby( ser.index.month ).mean()

    N=3 # 3 identical years to approx. cyclic average
    mmmi=np.concatenate([(30*(dfm.index.values-1)+360*n)
                         for n in range(N)] )
    mmm =np.concatenate([dfm.values]*N ) # 36 months of data
    
    # expand to daily, with a 360-day year
    resamp=np.interp(np.arange(0,360*N),
                     mmmi,mmm)
    lp=lp_filter.lowpass(resamp,dt=1,cutoff=30)
    clim=lp[360:720]
    
    def fit(times,clim=clim):
        if not isinstance(times,pd.Index):
            times=pd.DatetimeIndex(times)
        doy=times.dayofyear
        return np.interp( doy*360./365.25,
                          np.arange(360),clim)
    return fit

## 


# output synthesis is daily for this entire period.
# overkill, but there are some interesting features which
# wouldn't be resolved at monthly timescales
start_date=pd.Timestamp('2000-01-01')
end_date  =pd.Timestamp('2015-10-01')
expand=pd.date_range(start_date,end_date,freq='D')


def synth(df,name):
    
    # Site-specific modifications:
    #   Napa: generally no flow in summer, but in the dataframe this
    #   is just missing data, not 0 flow.
    if name=='napa':
        df=add_summer_noflow(df)

    syndf=pd.DataFrame({'Date':expand}).set_index('Date')

    for col in ['flow mgd','flow ft3/s',
                'NO3 mg/L N','NH3 mg/L N','PO4 mg/L P',
                'NO2 mg/L N','SKN mg/L N','TDN mg/L N','TDP mg/L P',
                'TKN mg/L N','TN  mg/L N','TP mg/L P','TRP mg/L P',
                'TSS mg/L','pH','temperature degC','urea* mg/L N']:
        if col not in df.columns:
            continue

        # Field specific portion:
        flag=col+" flag"
        # narrow to valid values for this field
        # col_df=pd.DataFrame( {col:df[ np.isfinite(df[col]) ][col] } )
        keep_cols=[col]
        if flag in df.columns:
            keep_cols.append( flag )
        rowsel=~df[col].isnull()
        col_df=df.loc[rowsel,keep_cols].copy()

        if len(col_df) == 0:
            continue

        # remove obvious outliers, depending on a priori knowledge
        if col=='temperature degC':
            valid=(col_df[col]>0) & (col_df[col]<35)
            col_df=col_df.loc[valid,:]

        # occasionally there are multiple measurements on the same day -
        # plus force the index to be dates, dropping the time so that
        # the join works below
        col_df=col_df.groupby(col_df.index.values.astype('datetime64[D]') ).mean()
        if flag in keep_cols:
            # should come in as an integer field, for which we assume that
            # negative values are unflagged.
            missing=(col_df[flag].isnull()) | (col_df[flag]<0)
            col_df.loc[missing,flag]=FLAG_ORIGINAL
        else:
            col_df[flag]=FLAG_ORIGINAL

        cmin,cmax=col_df[col].min(),col_df[col].max()

        # Add in climatology - choose one.
        # f_fit=seasonal_harmonic(col_df[col])
        # f_fit=monthly_climatology(col_df[col])
        f_fit=monthly_climatology_lp(col_df[col])

        # Choose gaps large enough to revert to climatology
        gap_days=60
        m=syndf.loc[:,()].join( col_df )

        if len(m)>len(syndf):
            print "Length of syndf: ",len(syndf)
            print "Length of join with col: ",len(m)
            print

            pdb.set_trace()
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
            # but make sure that doesn't create new extrema
            climate_col.loc[:,col] = climate_col[col].values.clip( cmin,cmax )


        df_new=pd.concat( [col_df,climate_col] )
        df_new=df_new.sort_index()

        # a little tricky - just joining here will make flag into a float.
        # assume that it's not that bad - the float is exact for integers 
        # up to 2^53 - i.e. much better than even an unsigned 32 bit int
        # still - cast back to int for the sake of netcdf and bitmasks
        syndf=syndf.join( df_new ) # maybe here is where flag gets floated?
        to_fill=np.isnan(syndf[col])
        syndf.loc[ np.isnan(syndf[col]),flag]=FLAG_INTERP
        # pandas 0.16 only fills in forward
        syndf[col].interpolate(inplace=True,method='linear')
        # so come back and copy backwards in time.
        syndf[col].fillna(method='backfill',inplace=True)
        syndf[flag] = syndf[flag].astype(np.int32)
    return syndf

## 

dest_dir='synth_inputs_v02'
os.path.exists(dest_dir) or os.mkdir(dest_dir)


csvs=glob.glob('compiled_inputs/*.csv')
csvs=[s for s in csvs if 'stormwater' not in s]

dfs=odict()
for fn in csvs:
    print fn
    name=os.path.basename(fn).replace('.csv','')

    dfs[name]=pd.read_csv(fn,parse_dates=['Date'],index_col='Date') 
    syn_df=synth(dfs[name],name)
    assert np.all( np.isfinite( syn_df.values ) )
        
    syn_df.to_csv(os.path.join(dest_dir,"%s.csv"%name))

## 

