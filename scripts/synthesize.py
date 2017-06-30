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
# CSV filenames 
site_names_files=['Davis_Ammonia', 'Davis_Flow', 'Davis_Nitrate', 'Davis_Nitrite+Nitrate',
			'Davis_Nitrite', 'Davis_Phosphorus', 'Manteca_Ammonia', 'Manteca_Flow',
			'Manteca_Nitrate', 'Manteca_Nitrite+Nitrate', 'Manteca_Nitrite', 
			'SacramentoFreeportDischarge', 'SacramentoFreeportNutrients', 
			'SacramentoVeronaDischarge', 'SanJoaquinDischarge', 'SanJoaquinNutrients',
			'Tracy_Ammonia', 'Tracy_Flow', 'Tracy_Nitrate', 'Tracy_Nitrite+Nitrate',
			'Tracy_Nitrite', 'Tracy_Phosphorus', 'Tracy', 'RegionalSan', 'Stockton']
			
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
		 
#FLAG_LOADING_STUDY=1
#FLAG_HDR=2
#FLAG_SUMMER_ZERO=4
#FLAG_SEASONAL_TREND=8
#FLAG_INTERP=16
#FLAG_MEAN=32
#FLAG_CLIPPED=64 # this one actually does get used as a bitmask.
#flag_bits=['LoadingStudy','HDR','Summer0','Trend','Interp','Mean','Clipped']		 


####################### Need to load in sites differently, since data for sites are split into various files


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
	
	
	