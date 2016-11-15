"""
This 
Standardize format of 'final effluent_concentrations_Mar2015.csv'
Write results to set of netcdf files.
"""
import os
import utils
import qnc
import pandas as pd
import periodictable as pt

## 
utils.path(os.path.join(os.environ['HOME'],'src'))

from pasticcio import P

ds=P.catalog.dataset(local_name='nutrient_model_loads_sources')

## 

# path to POTW data files
nut_data_path=os.path.join(os.environ['HOME'],'Google/1_Nutrient_Share/2_Data_NUTRIENTS/')

compiled=os.path.join(nut_data_path,'POTW Data/Effluent Characterization Data Yr2/')

# TRP is the last column, and occasionally holds the advertised data, but also
# holds comments.

all_recent=pd.read_csv(os.path.join(compiled,'final effluent_concentrations_Mar2015.csv'),
                       parse_dates=['Date'],
                       na_values=['#DIV/0!'],
                       dtype={'TRP (mg/L)':object})

trps=[] ; notes=[]

for val in all_recent['TRP (mg/L)'].values:
    try:
        trps.append(float(val))
        notes.append('')
    except ValueError:
        trps.append(np.nan)
        notes.append(val)

all_recent['TRP (mg/L)']=trps
all_recent['notes']=np.array(notes)

## 

facilities=[
['City of American Canyon','american_canyon'],
['City of Benicia WWTP','benicia'],
['City of Calistoga WWTP','calistoga'],
['Central Contra Costa Sanitary District','cccsd'],
['Chevron Richmond Refinery','chevron'],
['City of Burlingame WWTF','burlingame'],
['City of St. Helena Waste Water Treatment Plant','st_helena'],
['City of Sunnyvale','sunnyvale'],
['City of Palo Alto RWQCP','palo_alto'],
['City of Petaluma','petaluma'],
['Central Marin Sanitation Agency','marin_central'],
['Delta Diablo Sanitation District','delta_diablo'],
['EBDA','ebda'],
['East Bay Municipal Utility District','ebmud'],
['Fairfield - Suisun Sewer District','fairfield'],
['Pinole-Hercules WPCP','pinole_hercules'],
['Las Gallinas Valley Sanitary District','las_gallinas'],
['City of Millbrae WPCP','millbrae'],
['Mt. View Sanitary District','mt_view'],
['Novato Sanitary District','novato'],
['Napa Sanitation District','napa'],
['Phillip F. Meads Water Treatment Plant','meads'],
['Phillips 66 San Francisco Refinery','phillips66'],
['Rodeo Sanitary District','rodeo'],
['Sanitary District No.5 of Marin County Main Plant','marin_sd5_main'],
['Sanitary District No.5 of Marin County Paradise Cove Plant','marin_sd5_paradise_cove'],
['City of San Mateo','san_mateo'], 
['Sewerage Agency of Southern Marin','southern_marin'],
['Sausalito - Marin City Sanitary District','sausalito'],
['San Francisco International Airport - MLTP','sfo'],
['Shell Martinez Refinery','shell_martinez'],
['San Jose/Santa Clara Water Pollution Control Plant','san_jose'],
['Southeast Water Pollution Control Plant CCSF','sfpuc_southeast'],
['South San Francisco-San Bruno Water Quality Control Plant','san_bruno'],
['Sonoma Valley County Sanitation District','sonoma_valley'],
['Silicon Valley Clean Water','silicon_valley'],
['Treasure Island Water Pollution Control Plant','treasure_island'],
['Valero Refining Company - CA','valero'],
['Vallejo Sanitation & Flood Control District','vallejo'], 
['West County Agency','west_county'],
['Town of Yountville','yountville'], 
['Tesoro Golden Eagle Refinery','tesoro']
]


path=ds.local_path('data/final_effluent_conc_Mar2015')
os.path.exists(path) or os.makedirs(path)

for fac_long,fac_short in facilities:
    #if fac_short!='ebda':
    #    continue
    nc_fn=os.path.join(path,'%s.nc'%fac_short)

    print "%s => %s"%(fac_long,nc_fn)

    sel_facility=(all_recent['Facility']==fac_long)

    df=all_recent[sel_facility]

    nc=qnc.empty(fn=nc_fn,overwrite=True)

    mgd_to_m3s=0.043812636
    nc['time']['time']=df['Date']
    nc['flow']['time']=mgd_to_m3s*df['Flow  (MGD)']
    nc.flow.units='m3/s'
    nc.flow.method='average'
    nc.flow.long_name='flow'

    nc['peak_flow']['time']=mgd_to_m3s*df['Peak Flow (MGD)']
    nc.peak_flow.units='m3/s'
    nc.peak_flow.method='daily maximum'
    nc.peak_flow.long_name='daily peak flow'

    # for analytes which are a measure of a particular element, convert
    # to umol/L
    nc['TN']['time']=(1000./pt.N.mass)*df['TN (mg/L)']
    nc.TN.long_name='total nitrogen'
    nc.TN.units='umol/L'

    nc['TDN']['time']=(1000./pt.N.mass)*df['TDN (mg/L)']
    nc.TDN.long_name='total dissolved nitrogen'
    nc.TDN.units='umol/L'

    nc['TKN']['time']=(1000./pt.N.mass)*df['TKN (mg/L)']
    nc.TKN.long_name='total Kjeldahl nitrogen: organic N and ammonia'
    nc.TKN.units='umol/L'

    nc['SKN']['time']=(1000./pt.N.mass)*df['SKN (mg/L)']
    nc.SKN.long_name='soluble Kjeldahl nitrogen'
    nc.SKN.units='umol/L'

    # assuming that this mg/L is actually mg/L N
    nc['NO3']['time']=(1000./pt.N.mass)*df['NO3 (mg/L)']
    nc.NO3.long_name='nitrate'
    nc.NO3.units='umol/L'

    nc['NO2']['time']=(1000./pt.N.mass)*df['NO2 (mg/L)']
    nc.NO2.long_name='nitrite'
    nc.NO2.units='umol/L'

    nc['NH3']['time']=(1000./pt.N.mass)*df['Total NH3 (mg/L)']
    nc.NH3.long_name='total NH3'
    nc.NH3.units='umol/L'

    # might be more of an assumption here...
    nc['urea']['time']=(1000./pt.N.mass)*df['Urea* (mg/L)']
    nc.urea.long_name='urea'
    nc.urea.units='umol/L'

    nc['TP']['time']=(1000./pt.P.mass)*df['TP (mg/L)']
    nc.TP.long_name='total phosphorus'
    nc.TP.units='umol/L'

    nc['TDP']['time']=(1000./pt.P.mass)*df['TDP (mg/L)']
    nc.TDP.long_name='total dissolved phosphorus'
    nc.TDP.units='umol/L'

    nc['DRP']['time']=(1000./pt.P.mass)*df['DRP** (mg/L)']
    nc.DRP.long_name='dissolved reactive phosphorus'
    nc.DRP.units='umol/L'

    nc['TSS']['time']=df['TSS (mg/L)']
    nc.TSS.long_name='total suspended solids'
    nc.TSS.units='mg/L'

    nc['TRP']['time']=(1000./pt.P.mass)*df['TRP (mg/L)']
    nc.TRP.long_name='total reactive phosphorus'
    nc.TRP.units='umol/L'

    # pandas wants to just call these objects
    nc['notes']['time']=df['notes'].values.astype('S')

    # constants for the site
    nc['subembayment']['site']=[df['Subembayment'].iloc[0]]

    # Would be nice to bring in lat/lon, too.
    # maybe CF convention metadata so it's clear it's a stationary
    # time series.

    # Add in some derived fields:
    # from 'June 2015 progress update_final.rotated.Oct142015.pdf'
    #   at most discharges, o-PO4 is the dominant form of P.
    # 
    
    nc.close()
