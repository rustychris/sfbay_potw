"""
Sites have varying mixtures NO3, NO2 and "NOx" measurements.

Ostensibly, NOx=NO3 + NO2.

This is not always seen in the data, due to measurement noise 
but more often due to disparate data sources.

This script tries to make the 3 values consistent, and fill in
gaps where a reasonable guess can be made.

It's ugly.

A more complete solution would be to go back to source data and
identify the root cause of discrepancies.  Maybe when someone
has 2 months to spend analyzing load data...
"""

import numpy as np

def make_nitrogen_consistent(ds):
    for site in ds.site.values:
        for method in ['_conc','_load']:
            no3='NO3'+method
            no2='NO2'+method
            nox='NOx'+method

            # How to fill in the gaps between NO3, NOx, and NO2 data
            # First, try to look at it just per site.
            ds_site=ds.sel(site=site)

            # define in terms of the fraction of NOx which is NO3, which must
            # be in [0,1], and tends to 0.9 or so.

            no3_data=ds_site[no3].values.copy()
            no2_data=ds_site[no2].values.copy()
            nox_data=ds_site[nox].values.copy()

            # Some sites have bogus NOx data of 0
            # for times when no3 was nonzero.
            # sonoma_valley: looks like old data is NO3, but HDR is NOx, and HDR
            #  reports a lot of zeros.
            def ifnan(a,b):
                return np.where(np.isfinite(a),a,b)

            bad_nox=np.isfinite(nox_data) & (nox_data==0.0) & (ifnan(no3_data,-1)>0.0)
            if np.any(bad_nox):
                print("Site %s: false zeros in NOx will be turned to nan"%site)
                nox_data[bad_nox]=np.nan

            # and the opposite occurs, too. Assume that a nonzero nox requires a nonzero
            # no3.
            bad_no3=np.isfinite(no3_data) & (no3_data==0.0) & (ifnan(nox_data,-1)>0.0)
            if np.any(bad_no3):
                print("Site %s: false zeros in NO3 will be turned to nan"%site)
                no3_data[bad_no3]=np.nan

            def safe_divide(a,b):
                sel=np.isfinite(a+b) & (ifnan(b,0)>0)
                return a[sel]/b[sel]

            with np.errstate(divide='ignore'):
                ratio_samples_A=safe_divide( no3_data, nox_data)
                ratio_samples_B=1.0 - safe_divide( no2_data, nox_data )
                ratio_samples_C=safe_divide( no3_data, no3_data+no2_data )
            samples=np.concatenate( [ratio_samples_A,
                                     ratio_samples_B,
                                     ratio_samples_C] )
            samples=samples.clip(0,1)
            samples=samples[np.isfinite(samples)]
            if len(samples):
                no3_to_nox=samples.mean()
                if len(samples)>1:
                    dev=np.std(samples)
                else:
                    dev=np.inf
                #print("Site: %25s  NO3:NOx: %.3f +- %.3f"%(site,no3_to_nox,dev))
            else:
                # punt
                no3_to_nox=0.975
                #print("Site: %25s  NO3:NOx: %.3f [default]"%(site,no3_to_nox))
            # Every site is coming up with something -- default is never used.

            def update_missing(dest,src):
                """ helper to update missing values in dest with src """
                sel=np.isfinite(src) & np.isnan(dest)
                dest[sel]=src[sel]

            # Cases where we have two of the three values:
            update_missing(nox_data,no2_data+no3_data)
            update_missing(no2_data,nox_data-no3_data)
            update_missing(no3_data,nox_data-no2_data)

            # Cases where we have one of the three values, but don't use
            # singular NO2 to scale up to get NO3 and NOx because the
            # errors could be really big
            update_missing(nox_data,no3_data*1.0/no3_to_nox)
            update_missing(no3_data,nox_data*no3_to_nox)
            # and any additional no2 that implies:
            update_missing(no2_data,nox_data-no3_data)

            # Finally, enforce known constraints in cases where there is enough data:
            valid=np.isfinite(nox_data+no3_data+no2_data)
            if np.any(valid):
                small_nox=1.05*nox_data[valid]<(no3_data+no2_data)[valid]
                big_nox  =0.95*nox_data[valid]>(no3_data+no2_data)[valid]
                if np.any( small_nox | big_nox ):
                    print("Site: %25s: %d nox too large   %d nox too small  %d within 5%%"%(site,
                                                                                            np.nansum(big_nox),
                                                                                            np.nansum(small_nox),
                                                                                            np.nansum(~big_nox&~small_nox)))
                    print("   %d NOx records  %d NO3  %d NO2"%(np.isfinite(ds_site[nox]).sum(),
                                                               np.isfinite(ds_site[no3]).sum(),
                                                               np.isfinite(ds_site[no2]).sum()))
                    if np.nansum(big_nox):
                        # How much too large are we talking?
                        print("   Average overshoot: %.2f"%( np.nanmean( (nox_data[valid][big_nox]/
                                                                          (no3_data+no2_data)[valid][big_nox]) )))
                    if np.nansum(small_nox):
                        # How much too large are we talking?
                        print("  Average undershoot: %.2f"%( np.nanmean( (nox_data[valid][small_nox]/
                                                                          (no3_data+no2_data)[valid][small_nox]) )))
            else:
                print("No valid combinations of nox/no2/no3 for %s"%site)

            # And assign that back to the dataset:
            no3_flag=ds_site[no3+'_flag'].values
            no2_flag=ds_site[no2+'_flag'].values
            nox_flag=ds_site[nox+'_flag'].values

            combined_flags=no3_flag | no2_flag | nox_flag
            no3_changed=np.isnan(ds_site[no3].values) | (ds_site[no3].values != no3_data)
            no2_changed=np.isnan(ds_site[no2].values) | (ds_site[no2].values != no2_data)
            nox_changed=np.isnan(ds_site[nox].values) | (ds_site[nox].values != nox_data)
            
            no3_flag[no3_changed]=combined_flags[no3_changed]
            no2_flag[no2_changed]=combined_flags[no2_changed]
            nox_flag[nox_changed]=combined_flags[nox_changed]

            ds_site[no3].values[:] = no3_data
            ds_site[no2].values[:] = no2_data
            ds_site[nox].values[:] = nox_data
