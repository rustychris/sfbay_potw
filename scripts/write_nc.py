import os
import wkb2shp
import numpy as np
import osr
import pdb
import pandas as pd
import proj_utils
import xarray as xr
import proj_utils
import matplotlib.pyplot as plt
## 

csv_dir="synth_inputs_v02" # location of CSV files for POTWs

flows_shp="/home/rusty/models/suntans/spinupdated/inflows-v07.shp"

## 

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
           

def fix_names(ds):
    # fix up units - 
    real_vars=[col
               for col in ds.variables.keys()
               if col!='time' and not col.endswith('flag') ]

    for v in real_vars:
        pieces=v.split(' ')
        vnew=pieces[0]
        flagnew=vnew+"_flag"

        if v not in ds.variables: # DBG
            continue

        if len(pieces)==1:
            ds.rename({v+' flag':flagnew},inplace=True)
        else:
            vnew=pieces[0]
            ds.rename({v:vnew,v+' flag':flagnew},
                      inplace=True)
            ds[vnew].attrs['units']=pieces[1]
            if len(pieces)>2:
                rest=" ".join(pieces[2:])
        long_name=v
        for k in glossary:
            if long_name.startswith(glossary[k]):
                long_name=long_name.replace(k,"%s (%s)"%(k,glossary[k]))
                break
        if vnew in standards:
            ds[vnew].attrs['standard_name']=standards[vnew]
            print("  set standard name to %s"%ds[vnew].attrs['standard_name'])

        ds[vnew].attrs['long_name']=long_name
        ds[vnew].attrs['flags']=flagnew
        ds[flagnew].attrs['long_name']="Flags for %s"%vnew

        add_bitmask_metadata(ds[flagnew],
                             bit_meanings=['unknown','source_data','monthly_climatology',
                                           'interpolated','seasonal_zero'])


def add_bitmask_metadata(da,
                         bit_meanings=['zero','one','two','four',
                                       'eight','sixteen','thirtytwo']):
    """
    da: DataArray
    bit_meanings: 
    """
    assert( np.issubdtype(da.dtype.type,np.integer) )
    uniq_vals=np.unique(np.asarray(da))
    meanings=[]
    for val in uniq_vals:
        if val==0:
            meanings.append(bit_meanings[0])
        else:
            meaning = [m
                       for i,m in enumerate(bit_meanings[1:])
                       if val & (1<<i)]
            meanings.append( "_".join(meaning) )
    da.attrs['flag_values']=uniq_vals
    da.attrs['flag_meanings']=" ".join(meanings)


def fix_units(ds):
    for v in ds.variables.keys():
        if 'units' in ds[v].attrs:
            u_new=u_orig=ds[v].attrs['units']
            if u_orig=='mgd':
                u_new='Mgallon/day'
            elif u_orig=='degC':
                u_new='degree_Celsius'
            if u_new != u_orig:
                print "%s => %s"%(u_orig,u_new)
                ds[v].attrs['units']=u_new
            

xrs=[]

flows=wkb2shp.shp2geom(flows_shp)
u2l=proj_utils.mapper('EPSG:26910','WGS84')
utm=osr.SpatialReference()
utm.SetFromUserInput('EPSG:26910')

output_dir="nc_v03"
os.path.exists(output_dir) or os.makedirs(output_dir)

for flow_i,flow in enumerate(flows):
    if flow['driver']=='CSV':
        csv_fn,flow_field=flow['gages'].split('#')
        csv_path=os.path.join(csv_dir,csv_fn)
        df=pd.read_csv(csv_path,parse_dates=['Date'],index_col='Date')
        for field in df.columns:
            assert np.all( ~df[field].isnull() ) 
        ds=xr.Dataset.from_dataframe(df)
        # closer to standard:
        ds.rename({'Date':'time'},inplace=True)
        fix_names(ds)
        fix_units(ds)
        
        xy_utm = np.array(flow['geom'].coords).mean(axis=0)
        ll = u2l(xy_utm)

        # Tweaks to get ERDDAP to accept it as CF:
        # This is for the "orthogonal" multidimensional array representation
        # feeling around in the dark here...
        site_id=flow_i
        ds.coords['site']=[site_id]
        ds['site'].attrs['cf_role']='timeseries_id'
        ds['x_utm'] = xr.DataArray([xy_utm[0]],[ds.site])
        ds['y_utm'] = xr.DataArray([xy_utm[1]],[ds.site])
        ds['latitude'] = xr.DataArray([ll[1]],[ds.site])
        ds['longitude'] = xr.DataArray([ll[0]],[ds.site])
        ds.latitude.attrs['units']='degrees_north'
        ds.latitude.attrs['standard_name']='latitude_north'
        ds.longitude.attrs['units']='degrees_east'
        ds.longitude.attrs['standard_name']='longitude_east'
        ds.x_utm.attrs['units']='m'
        ds.y_utm.attrs['units']='m'
        ds.x_utm.attrs['standard_name']='projection_x_coordinate'
        ds.y_utm.attrs['standard_name']='projection_y_coordinate'
        ds.x_utm.attrs['_CoordinateAxisType']='GeoX'
        ds.y_utm.attrs['_CoordinateAxisType']='GeoY'
        ds.attrs['featureType']='timeSeries'

        ds['UTM10']=1
        ds.UTM10.attrs['grid_mapping_name']="universal_transverse_mercator"
        ds.UTM10.attrs['utm_zone_number']=10
        ds.UTM10.attrs['semi_major_axis']=6378137
        ds.UTM10.attrs['inverse_flattening'] = 298.257
        ds.UTM10.attrs['_CoordinateTransformType']="Projection"
        ds.UTM10.attrs['_CoordinateAxisTypes']="GeoX GeoY";
        ds.UTM10.attrs['crs_wkt']=utm.ExportToPrettyWkt()

        # add site to all the data variables, and some more CF 
        # attributes
        if 1:
            for v in ds.variables.keys():
                if v!='time' and ds[v].dims==('time',):
                    old=ds[v]
                    ds[v]=xr.DataArray([ds[v].values],[ds.site,ds.time])
                    ds[v].attrs=old.attrs

        # keep timebase consistent between files
        nc_path=os.path.join(output_dir,csv_fn.replace('.csv','.nc'))
        if 0:
            assert not os.path.exists(nc_path)
        else:
            os.path.exists(nc_path) and os.unlink(nc_path)
        encoding={'time':dict(units="seconds since 1970-01-01 00:00:00")}
        ds.to_netcdf(nc_path,encoding=encoding)
## 

!rsync -vzlP nc_v03/*.nc hpc.sfei.org:/opt/data/sfei/potw_flows/

