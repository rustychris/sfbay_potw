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

compile_dir="../outputs/intermediate/delta"
fig_dir="../outputs/figures"
output_dir="../outputs"

date_start=datetime.datetime(2000,1,1)
date_end  =datetime.datetime(2016,12,31)
dn_start=utils.to_dnum(date_start)
dn_end  =utils.to_dnum(date_end)
dns=np.arange(dn_start,dn_end+1)
fmt='%Y-%m-%d'

ds=xr.Dataset()
ds['time']=utils.to_dt64(dns)
ds['dnum']=('time',dns)
ds=ds.set_coords('dnum')

analytes=['flow',
          'NO3_conc', 'NO4_conc', 'NO2_conc', 'NN_conc', 'NH3_conc', 'NH4_conc', 'PO4_conc']
		  
# These match the names of the CSV files
site_names=['Davis', 'Manteca', 'Tracy', 'Stockton', 'RegionalSan', 'Sacramento', 'SanJoaquin']
ds['site']=( 'site', site_names)

			
# initialize full output array
for analyte in analytes:
    ds[analyte]=( ['time','site'],
             np.nan*np.ones( (len(ds.time),len(ds.site)) ) )

# set units for clarity upfront
ds.flow.attrs['units']='m3 s-1'
ds.NH3_conc.attrs['units']='mg/l N'
ds.NH4_conc.attrs['units']='mg/l N'
ds.NO3_conc.attrs['units']='mg/l N'
ds.NO2_conc.attrs['units']='mg/l N'
ds.NN_conc.attrs['units']='mg/l N'
ds.PO4_conc.attrs['units']='mg/l P'

# setup flag entries
for v in ds.data_vars.keys():
    ds[v+'_flag']=( ds[v].dims, np.zeros(ds[v].shape,'i2'))
    ds[v].attrs['flags']=v+'_flag'
		 

FLAG_SEASONAL_TREND=1
FLAG_INTERP=2
FLAG_MEAN=4
FLAG_CLIPPED=8 # this one actually does get used as a bitmask.
flag_bits=['Trend','Interp','Mean','Clipped']		 


# Read in Loading Study data via one csv per site
for site in ds.site: 
    site=site.item() # get to a str object
    # site_idx=list(ds.site).index(site) # 11

    csv=pd.read_csv(os.path.join(compile_dir,site+'.csv'),
                        parse_dates=['Date'])
    csv_dnums=utils.to_dnum(csv.Date)
    csv_date_i = np.searchsorted(dns,csv_dnums)
 
    # limit to the overlap between csv dates and output dates
    date_valid=(csv_dnums>=dns[0]) & (csv_dnums<dns[-1])
	
    # FLOW
	if 'flow mgd' in csv:
        flow=csv['flow mgd']
        valid=date_valid & (~flow.isnull().values)
        ds['flow'].sel(site=site)[csv_date_i[valid]] = flow[valid]
        flow_valid=valid 
	
        
    if 1: # NOx:
        nox=csv['NO3 mg/L N'].copy()
        try:
            nox += csv['NO2 mg/L N']
        except KeyError:
            print("No NO2 for %s - okay"%site)
        valid=date_valid & (~nox.isnull().values)

        ds['NOx_conc'].sel(site=site)[csv_date_i[valid]]=nox[valid]
        ds['NOx_conc_flag'].sel(site=site)[csv_date_i[valid]]=FLAG_LOADING_STUDY
        
    if 1: # NH3
        nh3=csv['NH3 mg/L N']
        valid=date_valid & (~nh3.isnull().values)

        ds['NH3_conc'].sel(site=site)[csv_date_i[valid]]=nh3[valid]
        ds['NH3_conc_flag'].sel(site=site)[csv_date_i[valid]]=FLAG_LOADING_STUDY
        conc_to_load('NH3',FLAG_LOADING_STUDY)
        
    if 1: # PO4
        po4=csv['PO4 mg/L P']
        valid=date_valid & (~po4.isnull().values)

        ds['PO4_conc'].sel(site=site)[csv_date_i[valid]]=po4[valid]
        ds['PO4_conc_flag'].sel(site=site)[csv_date_i[valid]]=FLAG_LOADING_STUDY
        conc_to_load('PO4',FLAG_LOADING_STUDY)	

def bin_mean(bins,values):
    sums=np.bincount(bins,weights=values)
    counts=np.bincount(bins)
    return sums/counts


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

	