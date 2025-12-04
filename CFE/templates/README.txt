These templates are used to switch between the data the LSE runs CFE on. 
- template.json: Runs on the Yalan data that is hardcoded into the set up
- aorc_template.json: Runs on AORC. AORC forcings includ observed flow, so only really need forcing_file.
- nldas_template.json: Runs on NLDAS. Flow is separate, so we need compare_results_file.
- mixed_template.json: Runs on AORC w/ NLDAS PET. I replaced compare_results_file with nldas forcings to grab. 
