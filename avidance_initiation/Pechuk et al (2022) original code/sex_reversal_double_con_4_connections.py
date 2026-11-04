import numpy as np
import supporting_functions as sf
import pandas as pd
from itertools import combinations
import os

job_ID = int(os.environ['LSB_JOBINDEX']) - 1
job_num = 2000

# =============================================================================
# Read parameters sets
# =============================================================================
Parameters_df = pd.read_csv("Parameters_combinations_df.csv", index_col = 0)

# =============================================================================
# Read csv of dimorphic responses for different stiluli strengths in hermaphrodites and males
# =============================================================================
sensory_strength_head = pd.read_csv("different_stimuli_samp_pol.csv")

# =============================================================================
# Set environment
# =============================================================================

# define neurons of interest:
circuit_neurons = {'ASH':'ASH[RL]', 'AVA':'AVA[RL]', 'AVD':'AVD[RL]', 'AVB':'AVB[RL]', 'PVC':'PVC[RL]', 'A':'[DV]A[0-9]', 'B':'[DV]B[0-9]'}
neurons = list(circuit_neurons.keys())

# time:
t = np.linspace(0,15,75001) # 15 seconds, in 0.0002 (0.2ms) jumps
correction_to_ms = int(round(0.001/(t[1]-t[0])))
sensory_start = 5000 * correction_to_ms
sensory_end = 10000 * correction_to_ms

# resting state: 
Vrest = np.reshape(np.repeat(np.array([-0.04]),len(neurons)), (len(neurons),1)) #units: V

# =============================================================================
# Check which connections are dimorphic based on the first parameters' set.
# Notice- the connectivity strength was changed between the sets but only for existing connection.
# Connections were not delited nor added between sets. 
# =============================================================================

# Adjacency matrix
Achem_herm = np.where(sf.str_to_nparray(sensory_strength_head.loc[0,"Schem_herm_adj"], (len(neurons), len(neurons))) != 0, 1, 0)
Agap_herm = np.where(sf.str_to_nparray(sensory_strength_head.loc[0,"Sgap_herm_adj"], (len(neurons), len(neurons))) != 0, 1, 0)
Achem_male = np.where(sf.str_to_nparray(sensory_strength_head.loc[0,"Schem_male_adj"], (len(neurons), len(neurons))) != 0, 1, 0)
Agap_male = np.where(sf.str_to_nparray(sensory_strength_head.loc[0,"Sgap_male_adj"], (len(neurons), len(neurons))) != 0, 1, 0)

# Find connections which are sex specific
herm_cells_chem, herm_ind_chem = sf.identify_dimorphic(Achem_herm, Achem_male, "chem", neurons, "herm")
herm_cells_gap, herm_ind_gap = sf.identify_dimorphic(Agap_herm, Agap_male, "gap", neurons, "herm")
male_cells_chem, male_ind_chem = sf.identify_dimorphic(Achem_herm, Achem_male, "chem", neurons, "male")
male_cells_gap, male_ind_gap = sf.identify_dimorphic(Agap_herm, Agap_male, "gap", neurons, "male")

# =============================================================================
# Unite all sex specific connections from all types (chem and gap)
# =============================================================================
herm_cells = []
herm_cells += sf.append_to_lists_start("chem", herm_cells_chem)
herm_cells += sf.append_to_lists_start("gap", herm_cells_gap)
cells_double_connections_herm = list(combinations(herm_cells, 2))
inds_double_connections_herm = [[[neurons.index(element) if element in neurons else element for element in cons] for cons in couple] for couple in cells_double_connections_herm]  

male_cells = []
male_cells += sf.append_to_lists_start("chem", male_cells_chem)
male_cells += sf.append_to_lists_start("gap", male_cells_gap)
cells_double_connections_male = list(combinations(male_cells, 2))
inds_double_connections_male = [[[neurons.index(element) if element in neurons else element for element in cons] for cons in couple] for couple in cells_double_connections_male]  

# =============================================================================
# Input that would trigger a specific voltage change:
# =============================================================================
# V = IR, I = V/R
sensory_input_calc = lambda res, volt: volt/res
sensory_voltage_changes = [75 * 10 **-3, 65 * 10 **-3, 55 * 10 ** -3, 45 * 10 **-3, 35 * 10 **-3, 25 * 10 ** -3, 15 * 10 **-3, 5 * 10 ** -3, 1 * 10 **-3]

# =============================================================================
# Cluster adjustments
# =============================================================================
job_segments = np.round(np.linspace(0, len(sensory_strength_head), job_num+1))
job_segments = job_segments.astype('int')
Dimorphic_double_con_Job = sensory_strength_head[job_segments[job_ID]:job_segments[job_ID+1]].copy()

# =============================================================================
# Initializing response arrays
# =============================================================================
names_dict_herm = [("_").join(cons[0] + cons[1]) for cons in cells_double_connections_herm]
names_dict_male = [("_").join(cons[0] + cons[1]) for cons in cells_double_connections_male]

# Responses
herm_to_male = sf.initialize_dimorphic_dict(names_dict_male, len(Dimorphic_double_con_Job), len(sensory_voltage_changes))
male_to_herm = sf.initialize_dimorphic_dict(names_dict_herm, len(Dimorphic_double_con_Job), len(sensory_voltage_changes))
herm_no_dimorphic = sf.initialize_dimorphic_dict(names_dict_herm, len(Dimorphic_double_con_Job), len(sensory_voltage_changes))
male_no_dimorphic = sf.initialize_dimorphic_dict(names_dict_male, len(Dimorphic_double_con_Job), len(sensory_voltage_changes))

# Check conditions
cond_herm_to_male = sf.initialize_dimorphic_dict(names_dict_male, len(Dimorphic_double_con_Job), 1)
cond_male_to_herm = sf.initialize_dimorphic_dict(names_dict_herm, len(Dimorphic_double_con_Job), 1)
cond_herm_no_dimorphic = sf.initialize_dimorphic_dict(names_dict_herm, len(Dimorphic_double_con_Job), 1)
cond_male_no_dimorphic = sf.initialize_dimorphic_dict(names_dict_male, len(Dimorphic_double_con_Job), 1)

for row in range(Dimorphic_double_con_Job.index[0],Dimorphic_double_con_Job.index[len(Dimorphic_double_con_Job)-1]+1):
    Sgap_herm = sf.str_to_nparray(Dimorphic_double_con_Job.Sgap_herm_adj[row], (len(neurons), len(neurons))) 
    Schem_herm = sf.str_to_nparray(Dimorphic_double_con_Job.Schem_herm_adj[row],  (len(neurons), len(neurons))) 
    Sgap_male = sf.str_to_nparray(Dimorphic_double_con_Job.Sgap_male_adj[row],  (len(neurons), len(neurons))) 
    Schem_male = sf.str_to_nparray(Dimorphic_double_con_Job.Schem_male_adj[row],  (len(neurons), len(neurons)))
    parameter_ind = Dimorphic_double_con_Job.parameters_index[row] 

# =============================================================================
# Current calculations
# =============================================================================
    for voltage in range(len(sensory_voltage_changes)):
        sensory_input = sensory_input_calc(Parameters_df.Rin[parameter_ind], sensory_voltage_changes[voltage])
        I = sf.input_current(Sgap_herm, t, range(sensory_start, sensory_end), neurons.index("ASH"), sensory_input, 0, range(0,len(t)), neurons.index("AVA"), Parameters_df.I_AVA[parameter_ind], 0, range(0,len(t)), neurons.index("AVB"), Parameters_df.I_AVB[parameter_ind], 0, range(0,len(t)), neurons.index("PVC"), Parameters_df.I_PVC[parameter_ind], 0) # units: A

# =============================================================================
# Transferring the dimorphic connection to the opposite sex    
# =============================================================================
# Hermaphrodites to males
# Addition
        herm_to_male, cond_herm_to_male = sf.double_con_sex_reversal_pol_samp(herm_to_male, cond_herm_to_male, cells_double_connections_male, inds_double_connections_male, Schem_herm, Sgap_herm, Schem_male, Sgap_male, Dimorphic_double_con_Job, row, t, Vrest, neurons, sensory_voltage_changes, voltage, sensory_start, sensory_end, I, Parameters_df, parameter_ind, correction_to_ms)

# Removal
        herm_no_dimorphic, cond_herm_no_dimorphic = sf.double_con_sex_reversal_pol_samp(herm_no_dimorphic, cond_herm_no_dimorphic, cells_double_connections_herm, inds_double_connections_herm, Schem_herm, Sgap_herm, Schem_male, Sgap_male, Dimorphic_double_con_Job, row, t, Vrest, neurons, sensory_voltage_changes, voltage, sensory_start, sensory_end, I, Parameters_df, parameter_ind, correction_to_ms)
        
# Males to hermaphrodites
# Addition              
        male_to_herm, cond_male_to_herm = sf.double_con_sex_reversal_pol_samp(male_to_herm, cond_male_to_herm, cells_double_connections_herm, inds_double_connections_herm, Schem_male, Sgap_male, Schem_herm, Sgap_herm, Dimorphic_double_con_Job, row, t, Vrest, neurons, sensory_voltage_changes, voltage, sensory_start, sensory_end, I, Parameters_df, parameter_ind, correction_to_ms)

# Removal
        male_no_dimorphic, cond_male_no_dimorphic = sf.double_con_sex_reversal_pol_samp(male_no_dimorphic, cond_male_no_dimorphic, cells_double_connections_male, inds_double_connections_male, Schem_male, Sgap_male, Schem_herm, Sgap_herm, Dimorphic_double_con_Job, row, t, Vrest, neurons, sensory_voltage_changes, voltage, sensory_start, sensory_end, I, Parameters_df, parameter_ind, correction_to_ms)
        
        print("finished row "+str(row)+", sensory input "+str(voltage))

# =============================================================================
# Hermaphrodites to males 
# =============================================================================
# Addition
for cons1_name in herm_to_male.keys():
    Dimorphic_double_con_Job["conditions_masc_herm_"+ cons1_name] = cond_herm_to_male[cons1_name]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_double_con_Job["response_masc_herm_"+ cons1_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_to_male[cons1_name][:,sens_input]
        Dimorphic_double_con_Job["A_vs_B_masc_herm_"+ cons1_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_to_male[cons1_name][:,sens_input]> 0, 1, 0)

# Removal
for cons2_name in herm_no_dimorphic.keys():
    Dimorphic_double_con_Job["conditions_no_dimorphic_herm_"+ cons2_name] = cond_herm_no_dimorphic[cons2_name]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_double_con_Job["response_no_dimorphic_herm_"+ cons2_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_no_dimorphic[cons2_name][:,sens_input]
        Dimorphic_double_con_Job["A_vs_B_no_dimorphic_herm_"+ cons2_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_no_dimorphic[cons2_name][:,sens_input]> 0, 1, 0)

# =============================================================================
# Males to hermaphrodites
# =============================================================================
# Addition
for cons3_name in male_to_herm.keys():
    Dimorphic_double_con_Job["conditions_fem_male_"+ cons3_name] = cond_male_to_herm[cons3_name]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_double_con_Job["response_fem_male_"+ cons3_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_to_herm[cons3_name][:,sens_input]
        Dimorphic_double_con_Job["A_vs_B_fem_male_"+ cons3_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_to_herm[cons3_name][:,sens_input]> 0, 1, 0)

# Removal
for cons4_name in male_no_dimorphic.keys():
    Dimorphic_double_con_Job["conditions_no_dimorphic_male_"+ cons4_name] = cond_male_no_dimorphic[cons4_name]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_double_con_Job["response_no_dimorphic_male_"+ cons4_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_no_dimorphic[cons4_name][:,sens_input]
        Dimorphic_double_con_Job["A_vs_B_no_dimorphic_male_"+ cons4_name + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_no_dimorphic[cons4_name][:,sens_input]> 0, 1, 0)

Dimorphic_double_con_Job.to_csv("250422_dimorphic_double_con_ID_"+str(job_ID)+".csv")