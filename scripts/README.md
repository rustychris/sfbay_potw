A full build requires the scripts to be run in the correct order, which is
a manual process currently:


Initial compilation:
   compile_delta_at_confluence.ipynb
   
   compile_bay_potw.py
     Handles the steps of converting units and corrections
     to source data (primarily Loading Study data), and writing out an intermediate csv file.
     HDR data is handled as a separate file, rather than merged into each discharger's CSV.
   
   wwtp_sources.py
     Process source data for Delta WWTPs into standardized intermediate CSV files.
     
   usgs_data_loading.py
     Process source data from USGS stations into standardized intermediate CSV files.

Additional compilation, depending on earlier sources:
   sac_nut.py
     Additional Sacramento nutrients data, but this pulls in some Freeport data generated
     in usgs_data_loading.py.

Final synthesis and output:
   synthesize.py
     Read in intermediate and potentially incomplete data files, fill gaps, write out final
     compiled, uniform datasets as netCDF and XLS.
  

== Extra Scripts - to be culled

Not maintained or currently used, or in the case of the csv file, an extraneous output
no longer needed.

sites_hdr_to_local.csv
  Maps source names in the HDR dataset to the names used in the output.





