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

FLAG_ORIGINAL=1 # original dataset
FLAG_MONTHLY=2 # monthly climatology
FLAG_INTERP=4

# output synthesis is daily for this entire period.
# overkill, but there are some interesting features which
# wouldn't be resolved at monthly timescales
start_date=pd.Timestamp('2000-01-01')
end_date  =pd.Timestamp('2015-06-01')

#def synth(df):
df=dfs['pinole'] # sample

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


def moving_average_nearest(x,y,n):
    """ Very inefficient implementation!
    """
    out=np.zeros_like(y)
    for i in range(len(x)):
        dists=np.abs(x-x[i])
        choose=np.argsort(dists)[:n]
        out[i]=np.mean(y[choose]) 
    return out


# Add in an adjustment for trends across gaps:
if 1:
    adjustment=col_df[col].values - f_fit(col_df.index)
    if 0:
        # this gets confused by large gaps with a trend across
        # the gap.
        adjustment=lp_filter.lowpass_fir(adjustment,winsize=5)
    else:
        adjustment=moving_average_nearest(utils.dt64_to_dnum(col_df.index.values),
                                          adjustment,
                                          n=3)
    adj_to_fit=np.interp( utils.dt64_to_dnum(climate_dates.values),
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
ax.plot_date(syndf.index,
             syndf[col],"k-")
sel=syndf[flag]!=FLAG_INTERP
ax.scatter(syndf.index[sel],syndf.loc[sel,col],50,syndf.loc[sel,flag],lw=0)

ax.plot(syndf.index,f_fit(syndf.index),'g-')
# ax.plot(syndf.index,err_interp,color='purple')
ax.plot(col_df.index,climate_err,color='purple')

ax.plot(col_df.index,adjustment,color='m')
ax.plot(syndf.index,valid,'mo')

## --
# harmonic fit to seasonal pattern:

if 1: # harmonic decomposition to get seasonal pattern
    f_fit=seasonal_harmonic(col_df[col])
    fit=f_fit(syndf.index)
elif 0: # periodic low pass
    dns=utils.dt64_to_dnum(col_df.index.values)
    expand_dns=utils.dt64_to_dnum(syndf.index.values)
    values=col_df[col].values
    period=365.25
    dn_mod=dns % period
    order=np.argsort(dn_mod)
    one_period=np.array( [dn_mod,values] ).T[order]
    N=10
    periods=np.concatenate( [one_period + [y*period,0] 
                             for y in range(N)] )
    resamp_dn=np.arange(0,N*period)
    resamp_data=np.interp(resamp_dn,periods[:,0],periods[:,1])
    lp=lp_filter.lowpass(resamp_data,cutoff=90.,dt=1.)
    fit=np.interp(5*period + expand_dns%period,
                  resamp_dn,lp)
    plt.figure(10).clf()
    plt.plot(resamp_dn,lp)
    plt.plot(resamp_dn,resamp_data)
else: # could do something simple based on monthly average
    assert False # not implemented

# the influence of a single observation is still too great.
# 

col_dns=utils.dt64_to_dnum(col_df.index.values)
fit_at_obs=harm_decomp.recompose(col_dns,comps,omegas)
err_at_obs=col_df[col].values - fit_at_obs
err_interp=np.interp(expand_dns,col_dns,err_at_obs)
adjust_fit=fit + err_interp


plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
ax.plot_date(syndf.index,
             syndf[col],"k-")
sel=syndf[flag]!=FLAG_INTERP
ax.scatter(syndf.index[sel],syndf.loc[sel,col],50,syndf.loc[sel,flag],lw=0)

ax.plot(syndf.index,fit,'g-')
ax.plot(syndf.index,adjust_fit,'r-')
ax.plot(syndf.index,err_interp,color='orange')

# that looks kind of bad...
# the monthly climatology numbers really stick out.
# better to do this in numpy and line things up better.

## 
import harm_decomp


col_df=pd.DataFrame( {col:df[ np.isfinite(df[col]) ][col] } )

# numpy approach:
obs_vals=col_df[col].values
obs_dns=utils.dt64_to_dnum(col_df.index.values)

omegas=np.arange(3) * 2*np.pi/365.25
comps=harm_decomp.decompose(obs_dns,obs_vals,omegas)
expand_dns=utils.dt64_to_dnum(syndf.index.values)
fit=harm_decomp.recompose(expand_dns,comps,omegas)

fit_at_obs=harm_decomp.recompose(col_dns,comps,omegas)
err_at_obs=obs_vals - fit_at_obs

# Update the fit to reflect multi-year trends
# problem here that a single observation can skew
# the error for too long.
if 0: # basic linear interpolation:
    err_interp=np.interp(expand_dns,col_dns,err_at_obs)
    err_interp_lp=lp_filter.lowpass_fir(err_interp,winsize=400)
else: # KDE style interpolation
    weights=np.zeros_like(expand_dns)
    values =np.zeros_like(expand_dns)
    for dn,err in zip(obs_dns,err_at_obs):
        sigma=50
        if 0: # gaussian
            shape=np.exp( -(expand_dns-dn)**2/(sigma**2) )
        else: 
            shape=1./(0.05+np.abs(expand_dns-dn)**2.5)
        weights+=shape
        values+=shape*err
    err_interp=values/weights
    err_interp_lp=err_interp


if 1:
    cweights=np.zeros_like(expand_dns)
    cvalues =np.zeros_like(expand_dns)
    for dn,val in zip(obs_dns,obs_vals):
        sigma=50
        if 0: # gaussian
            shape=np.exp( -(expand_dns-dn)**2/(sigma**2) )
        else: 
            shape=1./(0.05+np.abs(expand_dns-dn)**3)
        cweights+=shape
        cvalues+=shape*val
    # and try just feeding the climatology in as a constant, distance
    # source
    clim_shape=0.0000001*np.ones_like(cweights)
    cweights+=clim_shape
    cvalues+=clim_shape*fit
    fit3=cvalues/cweights

fit2=fit+err_interp_lp


plt.figure(2).clf()
fig,ax=plt.subplots(1,1,num=2)
ax.plot_date(obs_dns,obs_vals,'b-o')
ax.plot_date(expand_dns,fit,'g-')
ax.plot_date(expand_dns,fit2,'r-')
ax.plot_date(expand_dns,fit3,'k-')
ax.plot_date(obs_dns,err_at_obs,'-',color='orange')
ax.plot_date(expand_dns,err_interp,'-',color='purple')

ax.axis( (731503.01297117618,
          734859.24858847028,
          -1.6360691660457918,
          6.1153968375089036) )


# looking for these characteristics:
#    follow multi-year trends during the data
#    assume zero trend before/after data - i.e. extrapolation of trend
#    go through all data points
#    revert to climatology


## 

# what about a 2D kriging approach? might end up feeling too ad-hoc,
# since we have to force a weird unit conversion
obs_dns=utils.dt64_to_dnum(col_df.index.values)
# map dns to year, dayofyear

obs_year=col_df.index.year.astype(np.float64)
obs_doy=col_df.index.dayofyear.astype(np.float64)
obs_vals=col_df[col].values


plt.figure(12).clf()
fig,ax=plt.subplots(1,1,num=12)

ax.scatter(obs_doy,obs_year,40,obs_vals)

## 

# try again, but with geostatsmodels.
from geostatsmodels import utilities, variograms, model, kriging, geoplot
import pandas

P = np.array( [obs_doy,obs_year-2000,obs_vals] ).T
day_to_year=0.15
P[:,0] *= day_to_year # make one year about the same as the 15 years.  tune this.


# plt.figure(20).clf()
# plt.scatter( P[:,0], P[:,1], c=P[:,2], cmap=geoplot.YPcmap )
# plt.colorbar()
xmin, xmax = P[:,0].min(), P[:,0].max()
ymin, ymax = P[:,1].min(), P[:,1].max()
# plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
# for i in range( len( P[:,2] ) ):
#     x, y, por = P[i]
#     if( x < xmax )&( y > ymin )&( y < ymax ):
#         plt.text( x, y, '{:4.2f}'.format( por ) ) 
# plt.scatter( pt[0], pt[1], marker='x', c='k' )
# plt.text( pt[0]+100 , pt[1], '?')
# plt.xlabel('Day of year')
# plt.ylabel('Year') 

tolerance = 0.1
lags = np.arange( tolerance, 20, tolerance*2 )

plt.figure(21).clf()
# have to tune this:
if 0: 
    sill = np.var( P[:,2] ) # 1.6 is my addition
    svm = model.semivariance( model.spherical, ( 10, sill ) )
elif 0: # maybe
    sill = 1.6*np.var( P[:,2] ) # 1.6 is my addition
    svm = model.semivariance( model.linear, ( 14, sill ) )
elif 1: # maybe
    sill = 1.6*np.var( P[:,2] ) # 1.6 is my addition
    svm = model.semivariance( model.exponential, ( 17, sill ) )
    covfct = model.covariance( model.exponential, ( 17, sill ) )
elif 0:  # no.
    sill = np.var( P[:,2] ) 
    svm = model.semivariance( model.gaussian, ( 14, sill ) )
elif 0: # no.
    sill = np.var( P[:,2] ) 
    svm = model.semivariance( model.nugget, ( 14, sill ) )
elif 0: # meh.
    sill = np.var( P[:,2] ) 
    svm = model.semivariance( model.power, ( 0.25, 0.1 ) )
    

geoplot.semivariogram( P, lags, tolerance,model=svm,fignum=21 )


x=np.linspace(0,365.25*day_to_year,24)
y=np.arange(ymin-4,ymax+4,1.)
X,Y=np.meshgrid(x,y)
XY=np.array( [X.ravel(),Y.ravel()] ).T

#-# 
est, kstd = kriging.krige( P, covfct, XY, 'simple', N=16 )

est=est.reshape( X.shape )
kstd=kstd.reshape( X.shape )

## 
plt.figure(20).clf()
import gmtColormap
cmap=gmtColormap.viridis

img=plt.imshow(est,extent=[x[0]/day_to_year,x[-1]/day_to_year,y[0],y[-1]],
               interpolation='nearest',aspect='auto',
               cmap=cmap,vmin=2.3,vmax=5,origin='bottom')
scat=plt.scatter(P[:,0]/day_to_year,P[:,1],40,P[:,2],cmap=cmap,vmin=2.3,vmax=5)

plt.xlabel('Day of year')
plt.ylabel('Year')

plt.gcf().tight_layout()
