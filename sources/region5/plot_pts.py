import re
import glob
import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import logging
import numpy as np
log=logging.getLogger('pts')


## 

pts_dir='Sacramento_PointSource_Files'

pts_files=glob.glob(os.path.join(pts_dir,'*.*'))
pts_files.sort() # 159 of them.

## 

def read_pts(fn):
    ds=xr.Dataset()

    with open(fn,'rt') as fp:
        version,val=fp.readline().split()
        ds.attrs['version']=int(val)

        if ds.version!=3:
            log.warning("Expected version 3?")

        s = fp.readline()
        if 0: # too rigid
            lat=s[13:19]
            lon=s[32:40]
            name=s[40:].strip()
        else:
            m=re.match('^Latitude:\s*([-0-9\.]+)\s*Longitude:\s*([-0-9\.]+)([a-zA-Z].*)$',
                       s)
            lat=m.group(1)
            lon=m.group(2)
            name=m.group(3).strip()
        #
        ds['latitude']= ( ('source',), [float(lat)])
        ds['longitude']= ( ('source',), [float(lon)])
        ds['name']= ( ('source',), [name])

        dunno=fp.readline()

        headers=fp.readline()
    
    field_start=headers.index('M')
    field_count=int(headers[:field_start])

    matches=list(re.finditer('(^|\s)M', headers[field_start:]) )
    starts=[0] + [field_start+m.span()[0] for m in matches ]
    widths=np.diff(starts)
    widths[1]+=1 # because first one doesn't have the space in there.

    df=pd.read_fwf(fn,widths=widths,skiprows=3)

    datetimes=[ datetime.datetime.strptime(d,'%d%m%Y %H%M')
                for d in df.iloc[:,0] ]
    ds['date'] = ( ('date',), datetimes) #df.iloc[:,0] )
    
    for field in df.columns[1:].values:
        # drop the leading M
        ds[field[1:]] = ( ('date',), df.loc[:,field] )

    return ds

## 

dss=[]

for fn in pts_files:
    print fn
    dss.append( read_pts(fn) )

## 

lonlats=[ (ds.longitude.values,ds.latitude.values)
          for ds in dss ]
names=[ ds.name.values[0]
        for ds in dss ]
## 

from shapely import geometry
from stompy.spatial import wkb2shp


points=[ geometry.Point(ll[0],ll[1])
         for ll in lonlats]


wkb2shp.wkb2shp('region5-point_sources.shp',
                points,
                fields=dict(name=names),
                overwrite=True)

## 

# The useful ones are Davis, Sac Regional, UC Davis.  There is a Stockton East WD,
# but that appears to be a supplier of water.

ds_sac=dss[ names.index('SACRAMENTO REGIONAL WWTP') ]
ds_davis=dss[ names.index('Davis Wastewater Treatment Plant') ]
ds_ucdavis=dss[ names.index('UC DAVIS MAIN STP') ]
ds_woodland=dss[ names.index('WOODLAND WWTP') ]

## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

for ds in [ds_sac,ds_davis,ds_ucdavis,ds_woodland]:
    ax.semilogy(ds.date,ds.FLO,label=ds.name[0].values)
    
ax.legend(fontsize=8)
## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

for ds in [ds_sac,ds_davis,ds_ucdavis,ds_woodland]:
    ax.semilogy(ds.date,ds.NH4,label=ds.name[0].values)
ax.legend(fontsize=8)
ax.set_ylabel('NH4 load')

## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

for ds in [ds_sac,ds_davis,ds_ucdavis,ds_woodland]:
    ax.semilogy(ds.date,ds.NO3,label=ds.name[0].values)
ax.legend(fontsize=8)
ax.set_ylabel('NO3 load')
