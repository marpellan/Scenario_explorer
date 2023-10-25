# Scenarios Generator Model
## Introduction

The model aims to explore the evolution of the dwelling stock based on scenarios. It deals with new construction, renovation and demolition.

Once the surface are calculated, the model enables to calculate the operational and embodied GHGE year by year.

3 main files store the code (e.g. the created functions): 

- dynamic_epc_matrix
- generate_distribution_scenarios
- housing_needs 

and 3 main files are here to execute them and get the results :

- run_dynamic_epc_matrix
- run_generate_distribution_scenarios
- run_housing_needs 

## Dynamic_epc_matrix

### def update_pivot_table_reno
With a yearly renovation, return the targeted segments (e.g. dwelling type and EPC label) and an updated pivot table after renovation

### def update_pivot_table_dem
With a yearly demolition, return the targeted segments (e.g. dwelling type and EPC label) and an updated pivot table after demolition

### def update_surface_df
Return the repartition of the stock by EPC label after renovation and demolition

### def run_dynamic_simulation
Main function to run the 3 previous functions in a yearly basis and return the stock surface year by year by EPC label

### def calculate_jumps_epc_label
Group the renovation by jumps (e.g. EPC label before and after renovation) year by year

### def create_iamdf_reno
Create a iamdf (pyam) to group the renovation by "jumps"

### def create_iamdf_stock
Create a iamdf (pyam) of the updated stock year by year by EPC label

### def add_residential_dwelling_reno (silly function)
Add 'Residential_dwelling' as the sum of 'Collective_dwelling' and 'Individual_dwelling' in the iamdf_reno

### group_saut_dpe_reno (silly function)
Add the sum of saut, as the cumulative number of 1,2,3,4,5,6 jumps in EPC label 

### add_residential_dwelling_stock (silly function)
Add 'Residential_dwelling' as the sum of 'Collective_dwelling' and 'Individual_dwelling' in the iamdf_stock

### calculate_combined_embodied_reno_iamdf
Calculate embodied GHGE of renovations based on a iamdf_reno 
Returns the results by scenarios in MtCO2eq

### calculate_combined_operational_stock_iamdf
Calculate operational GHGE of the stock based on a iamdf_stock
Returns the results by scenarios in MtCO2eq

### group_dpe_stock (silly function)
Add cumulative GHGE as the sum of all EPC label in the result


## Generate_distribution_scenarios

### def generate_distribution
Generate a yearly distribution from a cumulated renovation potential
Distributions available are : 
- Constant
- Declining (start fast, then decline)
- Rising (start low, then rise)
- Normal (fastest at mid range in 2035)

## Housing_needs

### def calculate_housing_needs
Calculate housing needs in m2 based on population and m2/cap scenarios
Yearly results 

### def calculate_construction_needs_from_population_growth
Calculate growth in housing needs from t to t-1.
The growth can come from population growth and/or growth in demand (e.g. higher m2/cap)
Yearly results

### def run_construction_pyam
Calculate construction surface (in m2) and dwelling from :
- the growth in housing needs
- the demolition scenarios

Differentiate between collective and individual dwelling, which is subject to :
- % collective / individual in new constructed surface
- average m2 of new dwellings for collective and individual dwelling
Yearly results

### def calculate_embodied_new_construction
Calculate embodied GHGE of new construction based on construction and GHGE (e.g. carbon thresholds) scenarios

### def calculate_operational_new_construction
Calculate operational GHGE of new construction based on construction and GHGE (e.g. carbon thresholds) scenarios

