# create a time series matching the duration of the other (2000-2015)
# with tidal forcing for an ocean BC matchin the DFM grid.

# Copy the method from the sailtactics driver:
# all data in NAVD88.
import datetime
import pandas as pd
import utils

from gage_database import sfbay_gage_db
import forcing

## 

station_db = sfbay_gage_db.SFBayGages()

# will need to make this a composite gage, but for now, use PR directly
primary_tides = station_db.gage('Point Reyes','waterlevel')
secondary_tides = station_db.gage('Point Reyes','harmonic_waterlevel')

merged = forcing.MergeTidalTimeseriesFilter('Point Reyes failover tides',
                                            primary_tides,secondary_tides)
filtered = forcing.LowpassTimeseries(source=merged,
                                     cutoff_days = 90./(24*60),
                                     order=4)

adjusted = forcing.ShiftTimeseries(source=filtered,
                                   amplify=1.0,delay_s=0.0,offset=0.028)



## 

date_range=[datetime.datetime(2000,1,1),
            datetime.datetime(2015,6,1)]


data=adjusted.raw_data(*date_range)

## 

fields={'Date':utils.dnum_to_dt64(data[0]),
        'ssh m':data[1]}
        
df=pd.DataFrame(fields).set_index('Date')

## 

# Has some drift - would be nice to round this to 
# the true 6 minute intervals.

df.to_csv('ocean-tides-tmp.csv')

## 

