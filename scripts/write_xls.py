# Postprocessing step to copy netcdf outputs to an excel 
# spreadsheet.

import xarray as xr
import pandas as pd
import os

from stompy.spatial import wkb2shp, proj_utils

output_dir=os.path.join('..','outputs')

## 
ds=xr.open_dataset(os.path.join(output_dir,'sfbay_potw.nc'))

## 
if 1: 
    locs=wkb2shp.shp2geom(os.path.join('..', 'sources', 'discharge_approx_locations.shp') )

    df=pd.DataFrame(locs)

    df['utm_x']=[g.centroid.x
                 for g in df.geom]
    df['utm_y']=[g.centroid.y
                 for g in df.geom]

    utm=np.c_[ df.utm_x, df.utm_y]
    ll=proj_utils.mapper('EPSG:26910','WGS84')(utm)

    df['longitude']=ll[:,0]
    df['latitude']=ll[:,1]

    loc_df=df.loc[:, ['short_name', 'name', 'category', 'bc_type', 
                      'utm_x','utm_y',
                      'longitude', 'latitude',
                      'notes' ] ]
else:
    loc_df=None


writer = pd.ExcelWriter( os.path.join(output_dir,'sfbay_potw.xlsx'))

if loc_df is not None:
    loc_df.to_excel(writer,'locations',index=False)

# Break that out into one sheet per source
for site_name in nc.site.values:
    print site_name
    df=ds.sel(site=site_name).to_dataframe()

    df.to_excel(writer,site_name)
writer.save()

## 


# Need to fix the flag attributes in the netcdf (maybe they are set in the
# per-site data which goes to ERDDAP, but not in sfbay_potw.nc?)
# Seems that there are some deeper problems - wait and fix the flag situation
# when new Delta data comes in.
