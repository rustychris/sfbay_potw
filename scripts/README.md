compile_delta_at_confluence.ipynb

compile_bay_potw.ipynb
  Notebook - this single notebook handles the steps of converting units and corrections
  to source data (primarily Loading Study data), and writing out an intermediate csv file.
  Pretty sure this is actively used.  HDR data appears to be handled as a separate file, 
  rather than merged into each discharger's CSV.

data_overview_00.ipynb
master_post2012.py
plot_compiled.py
sites_hdr_to_local.csv
  Maps source names in the HDR dataset to the names used in the output.

synthesize.py
  Read in intermediate and potentially incomplete data files, fill gaps, write out final
  compiled, uniform datasets as netCDF and XLS.
  
wwtp_sources.py
  Process source data for Delta WWTPs into standardized intermediate CSV files.
  
usgs_data_loading.py
  Process source data from USGS stations into standardized intermediate CSV files.
  
synth_ocean_dfm.py





