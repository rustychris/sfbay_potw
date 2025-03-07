"""
Take loading data which may cover only a subset of the time period and/or
be missing many data points, synthesize a full data record, and write
netcdf and Excel outputs.
"""


from __future__ import print_function
import six
import numpy as np
import glob
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from osgeo import osr
from stompy import utils,filters
from stompy.spatial import wkb2shp, proj_utils

import nitrogen_utils

#

# Settings for file locations, time period
compile_dir="../outputs/intermediate"
fig_dir="../outputs/figures"
output_dir="../outputs"

date_start=datetime.datetime(2000,1,1)
date_end  =datetime.datetime(2016,12,31)

## 

# Constants for conversions
MGDtoCMS=0.043812636
CFStoCMS=0.028316847        


# Helper functions
def bin_mean(bins,values,empty=np.nan):
    sums=np.bincount(bins,weights=values)
    counts=np.bincount(bins)
    with np.errstate(all='ignore'):
        results=sums/counts
        results[counts==0]=empty
    return results


def mark_gaps(dnums,valid,gap_days,yearday_start=-1,yearday_end=367,include_ends=False):
    """
    for a timeseries, assumed to be dense and daily,
    return a mask which is true for gaps in valid data
    which span at least gap_days, limited to the portion of 
    the year given by yearday_start,yearday_end.
    include_ends: include the possibility of a gap of gap_days//2 at the beginning
    and end of the series (i.e. as if the next valid data point were very far off
    the end of the series)
    """
    doy=np.array([d - utils.dnum_jday0(d)
                  for d in dnums] )

    missing=~valid
    in_window=(doy>=yearday_start)&(doy<yearday_end)
    present=np.nonzero( ~missing | ~in_window)[0]

    mask=np.zeros( len(dnums),np.bool )

    for gstart,gend in zip( present[:-1],present[1:] ):
        if gend-gstart<gap_days:
            continue
        mask[ gstart+gap_days//2 : gend-gap_days//2 ] = True
        
    if include_ends:
        # too tired to think through the logic of how the ends combined with
        # the yeardays.
        assert yearday_start<0
        assert yearday_end>366
        first_gap=max(0,present[0]-gap_days//2)
        mask[:first_gap]=True
        final_gap=min( len(mask), present[-1]+gap_days//2 )
        mask[final_gap:]=True
    return mask

def add_summer_noflow(site,gap_days=45,day_start=100,day_end=305):
    """ Designed for Napa, but possibly extend to others.
    Gaps of more than gap_days, which fall within the period
    dayofyear between [day_start,day_end] are filled with zero 
    flow.
    """
    gap_mask = mark_gaps(dns, 
                         np.isfinite( ds['flow'].sel(site=site).values ),
                         gap_days=gap_days,
                         yearday_start=day_start,
                         yearday_end=day_end)
    ds.flow.sel(site=site).values[gap_mask] = 0
    ds.flow_flag.sel(site=site).values[gap_mask] = FLAG_SUMMER_ZERO

## 

# Initialize dataset, timeline, basic fields

dn_start=utils.to_dnum(date_start)
dn_end  =utils.to_dnum(date_end)
dns=np.arange(dn_start,dn_end+1)
fmt='%Y-%m-%d'

ds=xr.Dataset()
ds['time']=utils.to_dt64(dns)
ds['dnum']=('time',dns)
ds=ds.set_coords('dnum')

analytes=['flow',
          'NO3_conc', 'NO2_conc', 'NOx_conc', 'NH3_conc', 'PO4_conc', 'SiO2_conc']
		  
# These match the names of the CSV files
bay_site_names=['tesoro','american','sasm','novato','sunnyvale',
                'petaluma','rodeo','fs','valero','phillips66',
                'vallejo','ebmud','san_mateo','sfo','palo_alto','sausalito',
                'south_bayside','ddsd','burlingame','pinole','st_helena',
                'yountville','benicia','millbrae','sonoma_valley','napa',
                'cccsd','ebda','calistoga','central_marin','lg','west_county_richmond',
                'chevron','sf_southeast','shell','mt_view','marin5','san_jose',
                'south_sf','ch','treasure_island','false_sj','false_sac' ]
delta_site_names=['davis', 'manteca', 'tracy', 'stockton', 'sac_regional',
                  'sacramento_at_verona', 'san_joaquin_at_vernalis']
site_names=bay_site_names + delta_site_names

ds['site']=('site', site_names)

# initialize full output array
for analyte in analytes:
    ds[analyte]=( ['time','site'],
             np.nan*np.ones( (len(ds.time),len(ds.site)) ) )

# set units for clarity upfront
ds.flow.attrs['units']='m3 s-1'
ds.NH3_conc.attrs['units']='mg/l N'
ds.NO3_conc.attrs['units']='mg/l N'
ds.NO2_conc.attrs['units']='mg/l N'
ds.NOx_conc.attrs['units']='mg/l N'
ds.PO4_conc.attrs['units']='mg/l P'
ds.SiO2_conc.attrs['units']='mg/l Si'

# Add associated load variables for each of the analyte concentrations:
for v in ds.data_vars:
    if v.endswith('_conc'):
        vload=v.replace('_conc','_load')
        ds[vload]= ds[v].copy()
        ds[vload].attrs['units']='kg/day '+ds[v].attrs['units'].split()[1]

# Other analytes from previous versions:
# NH4: considered a synonym for NH3
# NN: a synonym for NOx

# setup flag entries
for v in ds.data_vars.keys():
    ds[v+'_flag']=( ds[v].dims, np.zeros(ds[v].shape,'i2'))
    ds[v].attrs['flags']=v+'_flag'
		 

FLAG_LOADING_STUDY=1
FLAG_HDR=2
FLAG_SUMMER_ZERO=4
FLAG_SEASONAL_TREND=8
FLAG_INTERP=16
FLAG_MEAN=32
FLAG_CLIPPED=64 # this one actually does get used as a bitmask.
FLAG_DELTA_DATA=128
FLAG_FLOW_INFERRED=256

# These may be joined by an underscore later on if multiple flags apply,
# so use camel-case to join words
flag_bits=['LoadingStudy','HDR','Summer0','Trend','Interp','Mean','Clipped','Delta',
           'FlowInferred']

## 

# Read in Loading Study data via one csv per site
for site in ds.site: 
    site=site.item() # get to a str object

    if site in bay_site_names:
        csv=pd.read_csv(os.path.join(compile_dir,site+'.csv'),
                        parse_dates=['Date'])
        src_flag=FLAG_LOADING_STUDY
    elif site in delta_site_names:
        csv=pd.read_csv(os.path.join(compile_dir,'delta',site+'.csv'),
                        parse_dates=['Date'])
        src_flag=FLAG_DELTA_DATA
    else:
        assert False

    # weed out bad timestamps
    sel=(csv.Date==csv.Date)
    csv=csv.loc[sel,:]
    
    csv_dnums=utils.to_dnum(csv.Date)
    csv_date_i = np.searchsorted(dns,csv_dnums)
 
    # limit to the overlap between csv dates and output dates
    date_valid=(csv_dnums>=dns[0]) & (csv_dnums<dns[-1])

    # helper function to set a subset of a variable's data, and
    # the flag values at the same time
    def set_with_flag(vname, values, valid, flag=src_flag):
        """ sets a subset of a variable in ds to the given values.
        flag is set to flag.  only values[valid] are assigned.
        vname is the string naming the variable
        """
        ds[vname].sel(site=site)[csv_date_i[valid]]=values[valid]
        ds[vname+"_flag"].sel(site=site)[csv_date_i[valid]]=src_flag
    
    # FLOW
    if 'flow ft3/s' in csv:
        # Convert cfs to m3/s
        flow=CFStoCMS * csv['flow ft3/s']
    elif 'flow mgd' in csv:
        # Convert mgd to m3/s
        flow=MGDtoCMS * csv['flow mgd']
    else:
        assert False
    flow_valid=date_valid & (~flow.isnull().values)
    set_with_flag('flow',flow,flow_valid)

    
    
    # NUTRIENTS
    if 'NO3 mg/L N' in csv:
        no3=csv['NO3 mg/L N']
        no3_valid=date_valid & (~no3.isnull().values)
        set_with_flag('NO3_conc',no3,no3_valid)

    if 'NO2 mg/L N' in csv:
        no2=csv['NO2 mg/L N']
        no2_valid=date_valid & (~no2.isnull().values)
        set_with_flag('NO2_conc',no2,no2_valid)
        
    if 'N+N mg/L N' in csv:
        nn=csv['N+N mg/L N']
        nn_valid=date_valid & (~nn.isnull().values)
        set_with_flag('NOx_conc',nn,nn_valid)

    if 'NH3 mg/L N' in csv:
        nh3=csv['NH3 mg/L N']
        nh3_valid=date_valid & (~nh3.isnull().values)
        set_with_flag('NH3_conc',nh3,nh3_valid)
        
    if 'PO4 mg/L P' in csv:
        po4=csv['PO4 mg/L P']
        po4_valid=date_valid & (~po4.isnull().values)
        set_with_flag('PO4_conc',po4,po4_valid)

## 

# Don't do this -- rather, base any extrapolation on the quantity
# that was measured, then only at the end do we change to loads
# if it started as concentrations

# # compute loads where there is flow and concentration:
# for fld in ['NO3_conc','NO2_conc','NOx_conc','NH3_conc', 'PO4_conc']:
#     flow_valid=ds['flow_flag'].values>0
#     conc_valid=ds[fld+'_flag'].values>0
#     load_valid=flow_valid&conc_valid
# 
#     load= (  ds[fld].values[load_valid] 
#              * ds['flow'].values[load_valid] )
#     #  ...  L/m3   s/day   kg/mg  
#     load *= 1e3  * 86400 * 1e-6
#     fld_load=fld.replace('_conc','_load')
#     ds[fld_load].values[load_valid]=load
#     ds[fld_load+'_flag'].values[load_valid]=ds[fld+'_flag'].values[load_valid]

##

# Bring in some additional information --

# Napa: special handling, typically no flow in summer
add_summer_noflow(site='napa',gap_days=45,day_start=100,day_end=305)

## HDR Data

# Load the HDR data in long format
hdr_fn=os.path.join(compile_dir,'hdr_parsed_long.csv')
hdr=pd.read_csv(hdr_fn)
month_starts=[ datetime.datetime(year=int(r.year),month=int(r.month),day=1)
               for ri,r in hdr.iterrows()]
hdr['dn_start']=utils.to_dnum( np.array(month_starts))

# bring in HDR data
sites=hdr.site.unique()

# Read mapping from sites_hdr_to_local.csv
hdr_name_map=dict([ [s.strip() for s in line.split(',')]
                    for line in open('sites_hdr_to_local.csv')])

# iterate over hdr names and their local ('loading study') equivalents
for hdr_name,ls_name in six.iteritems(hdr_name_map):
    if ls_name not in list(ds.site.values):
        continue
    print( "%25s => %-25s"%(hdr_name,ls_name))
    hdr_site = hdr[ hdr.site==hdr_name ]

    # move the analytes back to columns
    hdr_site = hdr_site.pivot(index='dn_start',columns='analyte',values='value').reset_index()
    dn_end=np.zeros(len(hdr_site.dn_start.values),'f8')
    dn_end[:-1]=hdr_site.dn_start.values[1:]
    dn_end[-1] = hdr_site.dn_start.values[-1] + 31
    hdr_site['dn_end']=dn_end
    hdr_site.head()

    for ri,r in hdr_site.iterrows():
        time_slc=slice(*np.searchsorted(ds.dnum,[r.dn_start,r.dn_end]))

        ds_site = ds.sel(site=ls_name)

        def from_hdr(ds_fld,hdr_fld,factor=1):    
            ds_site[ds_fld].values[time_slc] = r[hdr_fld] * factor
            ds_site[ds_fld+'_flag'].values[time_slc] = FLAG_HDR

        # overwrite with HDR data, constant over month
        from_hdr('flow','flow_mgd',MGDtoCMS) 
        from_hdr('NOx_load','NOx_kgN_per_day',1)
        from_hdr('NH3_load','ammonia_kgN_per_day',1)
        # unclear whether we should go with diss_OrthoP, or total_kgP ...
        from_hdr('PO4_load','diss_OrthoP_kgP_per_day')
        # TKN is also in there, but we're not yet worrying about it.

## 

# Read in the previously-composited San Joaquin River at Vernalis
# nutrient data
sj=pd.read_csv('../outputs/intermediate/delta/san_joaquin_composite_conc_zz.csv')
sj.rename(columns={sj.columns[0]:'Date'},inplace=True)
dates=[ datetime.datetime.strptime(d,'%Y-%m-%d')
        for d in sj.Date.values ]
sj_dnums=utils.to_dnum(np.array(dates))
ds_dnums=utils.to_dnum(ds.time)
flag=FLAG_INTERP # call it all interp, since the provenance is obscured at this point

sj_cols=['NH4','NO3','PO4','SiO2']
potw_cols=['NH3_conc','NO3_conc','PO4_conc','SiO2_conc']

for sj_col,potw_col in zip(sj_cols,potw_cols):
    ds[potw_col].sel(site='san_joaquin_at_vernalis').values[:] = np.interp(ds_dnums,
                                                                           sj_dnums,sj[sj_col].values)

## 

# total headache to work through making NOx, NO2 and NO3 self-consistent.
# This could use some real refactoring.

nitrogen_utils.make_nitrogen_consistent(ds)

## 

def fill_and_flag(ds,fld,site,
                  lowpass_days=3*365,
                  shortgap_days=45 # okay to interpolate a little over a month?
              ):
    """
    Update a single field for a single site in ds, by
    extracting long-term trends, seasonal cycle, and
    interpolating between these and measured data
    """
    # first, create mapping from time index to absolute month
    dts=utils.to_datetime(dns)
    absmonth = [12*dt.year + (dt.month-1) for dt in dts]
    absmonth = np.array(absmonth) - dts[0].year*12
    month=absmonth%12

    fld_in=ds[fld].sel(site=site)
    orig_values=fld_in.values
    fld_flag=ds[fld+'_flag'].sel(site=site)

    prefilled=fld_flag.values & (FLAG_SEASONAL_TREND | FLAG_INTERP | FLAG_MEAN)        
    fld_in.values[prefilled]=np.nan # resets the work of this loop in case it's run multiple times
    n_valid=np.sum(~fld_in.isnull())        

    if n_valid==0:
        msg=" --SKIPPING--"
    else:
        msg=""
    print("   field: %s  %d/%d valid input points %s"%(fld,n_valid,len(fld_in),msg))

    if n_valid==0:
        return

    # get the data into a monthly time series before trying to fit seasonal cycle
    valid = np.isfinite(fld_in.values)
    absmonth_mean=bin_mean(absmonth[valid],fld_in.values[valid])
    month_mean=bin_mean(month[valid],fld_in.values[valid])

    if np.sum(np.isfinite(month_mean)) < 12:
        print("Insufficient data for seasonal trends - will fill with sample mean")
        trend_and_season=np.nanmean(month_mean) * np.ones(len(dns))
        t_and_s_flag=FLAG_MEAN
    else:
        # fit long-term trend and a stationary seasonal cycle
        # this removes both the seasonal cycle and the long-term mean,
        # leaving just the trend
        trend_hf=fld_in.values - month_mean[month]
        lp = filters.lowpass_fir(trend_hf,lowpass_days,nan_weight_threshold=0.01)
        trend = utils.fill_invalid(lp)
        # recombine with the long-term mean and monthly trend 
        # to get the fill values.
        trend_and_season = trend + month_mean[month]
        t_and_s_flag=FLAG_SEASONAL_TREND

    # long gaps are mostly filled by trend and season
    gaps=mark_gaps(dns,valid,shortgap_days,include_ends=True) 
    fld_in.values[gaps] = trend_and_season[gaps]
    fld_flag.values[gaps] = t_and_s_flag

    still_missing=np.isnan(fld_in.values)
    fld_in.values[still_missing] = utils.fill_invalid(fld_in.values)[still_missing]
    fld_flag.values[still_missing] = FLAG_INTERP

    # Make sure all flows are nonnegative
    negative=fld_in.values<0.0
    fld_in.values[negative]=0.0
    fld_flag.values[negative] |= FLAG_CLIPPED

    if 0: # illustrative(?) plots
        fig,ax=plt.subplots()
        ax.plot(dns,orig_values,'m-o',label='Measured %s'%fld)
        ax.plot(dns,fld_in,'k-',label='Final %s'%fld,zorder=5)
        # ax.plot(dns,month_mean[month],'r-',label='Monthly Clim.')
        # ax.plot(dns,trend_hf,'b-',label='Trend w/HF')
        ax.plot(dns,trend,'g-',lw=3,label='Trend')
        ax.plot(dns,trend_and_season,color='orange',label='Trend and season')
            
## 
ds_fill=ds.copy(deep=True)

# Interpolate on conc, then calculate load via flow.
# For both NO3 and NH4, concentration is less variable than
# load in the majority of cases.
# That may be affected by previous steps in compile_bay_potw.py
# that impose constant concentrations, but several spot checks
# show that even outside those cases, concentration is slightly
# more constant.  This is counter to my expectation.
conc_fields=[s for s in ds_fill.data_vars if s.endswith('_conc')]

for site in ds_fill.site.values:
    fill_and_flag(ds_fill,'flow',site)

    flows=ds_fill.flow.sel(site=site)
    for conc_fld in conc_fields:
        load_fld=conc_fld.replace('_conc','_load')
        conc =ds[conc_fld].sel(site=site)
        load =ds[load_fld].sel(site=site)

        conc_valid=np.isfinite(conc.values)
        load_valid=np.isfinite(load.values)

        load_flags=ds.sel(site=site)[load_fld+'_flag'].values.copy()
        conc_flags=ds.sel(site=site)[conc_fld+'_flag'].values.copy()

        # load: kg/day
        # flow: m3/s
        # conc: mg/l
        # mg/l * m3/s ..... l/m3   s/day     kg/mg
        conc_to_load_factor=1e3  * 86400 * 1e-6
        load_to_conc_factor=1.0/conc_to_load_factor

        # when we have HDR data, let that override spot samples of
        # concentration since it is already a composite of several concentration
        # measurements
        hdr_loads=(load_flags & FLAG_HDR)>0
        combined_conc=np.where(conc_valid&(~hdr_loads),conc,
                               load/flows * load_to_conc_factor)

        # Flag the converted values as such
        ds_fill.sel(site=site)[conc_fld+'_flag'].values[load_valid&(~conc_valid)]=FLAG_FLOW_INFERRED
        ds_fill.sel(site=site)[conc_fld].values[:]=combined_conc
        fill_and_flag(ds_fill,conc_fld,site)
        
        # Now copy back to loads via flow.
        # Note that that this may overwrite load data if there was both a concentration measured
        # and a load "measured"
        
        ds_fill.sel(site=site)[load_fld].values[:]=ds_fill.sel(site=site)[conc_fld]*flows*conc_to_load_factor
        # This probably isn't correct
        # Loads which were valid, and not overwritten by a concentration retain their
        # original flag.
        # Any other datapoints get the 
        new_load_flags=(~load_valid)|conc_valid
        load_flags[new_load_flags] = FLAG_FLOW_INFERRED | conc_flags[new_load_flags]
        ds_fill.sel(site=site)[load_fld+'_flag'].values=load_flags



## 

# Mark the "types" of the sites (false, potw, refinery)
ds_fill['site_type']=('site',[' '*20]*len(ds_fill.site.values))

for s in ['tesoro','phillips66','valero','chevron','shell']:
    ds_fill.site_type.loc[s] = 'refinery'
for s in ['american','sasm','novato','sunnyvale','petaluma',
          'rodeo','fs','vallejo','ebmud','san_mateo','sfo',
          'palo_alto','sausalito','south_bayside','ddsd',
          'burlingame','pinole','st_helena','yountville',
          'benicia','millbrae','sonoma_valley','napa','cccsd',
          'ebda','calistoga','central_marin','lg','west_county_richmond',
          'sf_southeast','mt_view','marin5','san_jose',
          'south_sf','ch','treasure_island']:
    ds_fill.site_type.loc[s] = 'potw'
for s in ['false_sac','false_sj']:
    ds_fill.site_type.loc[s] = 'false'

for s in ['manteca','davis','tracy','stockton','sac_regional']: # Delta sources
    ds_fill.site_type.loc[s]='potw'
    
for s in ['sacramento_at_verona','san_joaquin_at_vernalis']:
    ds_fill.site_type.loc[s]='river'
    
for st in np.unique( ds_fill.site_type.values ):
    count =np.sum(ds_fill.site_type==st)
    print("Site type %s: %d"%(st,count))

##     

# assign lat/lon from approx_discharge_locations.shp
locs=wkb2shp.shp2geom('../sources/discharge_approx_locations.shp')

ds_fill['utm_x']=( ('site',), np.nan * np.ones(len(ds_fill.site)))
ds_fill['utm_y']=1*ds_fill.utm_x
# None is better than "" here, as it does not impose a field length
ds_fill['name']=( ('site',), [None]*len(ds_fill.site))

print("Discharges in discharge_approx_locations.shp, but not in sfbay_potw data")
for rec in locs:
    if rec['short_name'] not in ds_fill.site.values:
        print("    '%s'"%rec['short_name'])
        continue
    xy=np.array(rec['geom'].centroid)

    sel=dict(site=rec['short_name'])
    ds_fill['utm_x'].loc[sel]=xy[0]
    ds_fill['utm_y'].loc[sel]=xy[1]
    ds_fill['name'].loc[sel]=rec['name']

missing=ds_fill['site'][ np.isnan(ds_fill['utm_x'].values) ].values
if len(missing):
    print("%d sites in sfbay_potw data, but without a location from discharge_approx_locations.shp"%(len(missing)))
    print(",".join(missing))
else:
    print("All site in sfbay_potw matched with a lat/lon")

xy=np.c_[ ds_fill.utm_x, ds_fill.utm_y]
ll=proj_utils.mapper('EPSG:26910','WGS84')(xy)
ds_fill['latitude']=( ('site',),ll[:,1])
ds_fill['longitude']=( ('site',),ll[:,0])
ds_fill=ds_fill.set_coords(['utm_x','utm_y','latitude','longitude'])

##

# Try writing directly to a netcdf file that ERDDAP is willing to load:
# map some abbreviations to descriptive names
glossary={'TDN':'total dissolved nitrogen',
          'SKN':'soluble Kjeldahl nitrogen',
          'TDP':'total dissolved phosphorus',
          'TKN':'total Kjeldahlf nitrogen',
          'TP':'total phosphorus',
          'TSS':'total suspended solids'}

# And some standard names:
standards={'NO3':'mass_concentration_of_nitrate_in_sea_water',
           'NO2':'mass_concentration_of_nitrite_in_sea_water',
           'TSS':'mass_concentration_of_suspended_solids_in_sea_water',
           'temperature':'sea_water_temperature'}



def add_bitmask_metadata(da,
                         bit_meanings=['b1','b2','b4','b8','b16','b32','b64','b128']):
    """
    da: DataArray
    bit_meanings: 
    """
    assert( np.issubdtype(da.dtype.type,np.integer) )
    uniq_vals=np.unique(np.asarray(da))
    meanings=[]
    for val in uniq_vals:
        if val==0:
            meanings.append("unset")
        else:
            meaning = [m
                       for i,m in enumerate(bit_meanings)
                       if val & (1<<i)]
            meanings.append( "_".join(meaning) )
    da.attrs['flag_values']=uniq_vals
    da.attrs['flag_meanings']=" ".join(meanings)

if 1: # fix names, bitmask metadata
    # fix up units - 
    for v in ds_fill.data_vars.keys():
        if v.endswith('flag') or v in ['site_type','name']:
            continue

        # Newer code discards most of the weird things in glossary, but
        # this code is in place for the future.
        
        # Use glossary dict to write nicer long names
        long_name=v
        for k in glossary:
            if long_name.startswith(glossary[k]):
                long_name=long_name.replace(k,"%s (%s)"%(k,glossary[k]))
                break
        # Use standards dict to choose nice short names
        for k in standards:
            if v.startswith(k):
                ds_fill[v].attrs['standard_name']=standards[k]
                print("  set standard name to %s"%ds_fill[v].attrs['standard_name'])

        # Handle the flags field
        ds_fill[v].attrs['long_name']=long_name
        flag_name="%s_flag"%(v)
        if flag_name in ds_fill:
            ds_fill[v].attrs['flags']=flag_name
            ds_fill[flag_name].attrs['long_name']="Flags for %s"%v

            add_bitmask_metadata(ds_fill[flag_name],
                                 bit_meanings=flag_bits)

## 
# closer to standard:

utm=osr.SpatialReference()
utm.SetFromUserInput('EPSG:26910')

# Tweaks to get ERDDAP to accept it as CF:
# This is for the "orthogonal" multidimensional array representation
# feeling around in the dark here...
# site_id=flow_i
# ds_fill.coords['site']=[site_id]
ds_fill['site'].attrs['cf_role']='timeseries_id'
ds_fill.latitude.attrs['units']='degrees_north'
ds_fill.latitude.attrs['standard_name']='latitude_north'
ds_fill.longitude.attrs['units']='degrees_east'
ds_fill.longitude.attrs['standard_name']='longitude_east'
ds_fill.utm_x.attrs['units']='m'
ds_fill.utm_y.attrs['units']='m'
ds_fill.utm_x.attrs['standard_name']='projection_x_coordinate'
ds_fill.utm_y.attrs['standard_name']='projection_y_coordinate'
ds_fill.utm_x.attrs['_CoordinateAxisType']='GeoX'
ds_fill.utm_y.attrs['_CoordinateAxisType']='GeoY'
ds_fill.attrs['featureType']='timeSeries'

ds_fill['UTM10']=1
ds_fill.UTM10.attrs['grid_mapping_name']="universal_transverse_mercator"
ds_fill.UTM10.attrs['utm_zone_number']=10
ds_fill.UTM10.attrs['semi_major_axis']=6378137
ds_fill.UTM10.attrs['inverse_flattening'] = 298.257
ds_fill.UTM10.attrs['_CoordinateTransformType']="Projection"
ds_fill.UTM10.attrs['_CoordinateAxisTypes']="GeoX GeoY";
ds_fill.UTM10.attrs['crs_wkt']=utm.ExportToPrettyWkt()

##
# Output:


# keep timebase consistent between files
nc_path=os.path.join(output_dir,'sfbay_delta_potw.nc')
os.path.exists(nc_path) and os.unlink(nc_path)
encoding={'time':dict(units="seconds since 1970-01-01 00:00:00")}
ds_fill.to_netcdf(nc_path,encoding=encoding)

## 

# And write an xls file, too.  Reload from disk to ensure consistency.
ds_reload=xr.open_dataset(os.path.join(output_dir,'sfbay_delta_potw.nc'))

writer = pd.ExcelWriter( os.path.join(output_dir,'sfbay_delta_potw.xlsx'))

# Break that out into one sheet per source
for site_name in ds_reload.site.values:
    print(site_name)
    df=ds_reload.sel(site=site_name).to_dataframe()

    df.to_excel(writer,site_name)
writer.save()

## 

