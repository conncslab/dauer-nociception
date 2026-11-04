import numpy as np
import supporting_functions as sf
import pandas as pd
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
# Input that would trigger a specific voltage change:
# =============================================================================
# V = IR, I = V/R
sensory_input_calc = lambda res, volt: volt/res
sensory_voltage_changes = [75 * 10 **-3, 65 * 10 **-3, 55 * 10 ** -3, 45 * 10 **-3, 35 * 10 **-3, 25 * 10 ** -3, 15 * 10 **-3, 5 * 10 ** -3, 1 * 10 **-3]

# =============================================================================
# Cluster adjustments
# =============================================================================
job_segments = np.round(np.linspace(0, len(sensory_strength_head), job_num + 1))
job_segments = job_segments.astype('int')
Dimorphic_connections_Job = sensory_strength_head[job_segments[job_ID]:job_segments[job_ID+1]].copy()

# =============================================================================
# Initializing response arrays
# =============================================================================
names_dict_herm_chem = [herm_cells_chem[connection][0]+ "_to_"+ herm_cells_chem[connection][1] for connection in range(len(herm_cells_chem))]
names_dict_herm_gap = [herm_cells_gap[connection][0]+ "_to_"+ herm_cells_gap[connection][1] for connection in range(len(herm_cells_gap))]
names_dict_male_chem = [male_cells_chem[connection][0]+ "_to_"+ male_cells_chem[connection][1] for connection in range(len(male_cells_chem))]
names_dict_male_gap = [male_cells_gap[connection][0]+ "_to_"+ male_cells_gap[connection][1] for connection in range(len(male_cells_gap))]

#### Responses ####
# Adding conncetions
herm_to_male_chem = sf.initialize_dimorphic_dict(names_dict_male_chem, len(Dimorphic_connections_Job), len(sensory_voltage_changes))
herm_to_male_gap = sf.initialize_dimorphic_dict(names_dict_male_gap, len(Dimorphic_connections_Job), len(sensory_voltage_changes))
male_to_herm_chem = sf.initialize_dimorphic_dict(names_dict_herm_chem, len(Dimorphic_connections_Job), len(sensory_voltage_changes))
male_to_herm_gap = sf.initialize_dimorphic_dict(names_dict_herm_gap, len(Dimorphic_connections_Job), len(sensory_voltage_changes))

# Removing connections
herm_no_dimorphic_chem = sf.initialize_dimorphic_dict(names_dict_herm_chem, len(Dimorphic_connections_Job), len(sensory_voltage_changes))
herm_no_dimorphic_gap = sf.initialize_dimorphic_dict(names_dict_herm_gap, len(Dimorphic_connections_Job), len(sensory_voltage_changes))
male_no_dimorphic_chem = sf.initialize_dimorphic_dict(names_dict_male_chem, len(Dimorphic_connections_Job), len(sensory_voltage_changes))
male_no_dimorphic_gap = sf.initialize_dimorphic_dict(names_dict_male_gap, len(Dimorphic_connections_Job), len(sensory_voltage_changes))

#### Check conditions ####
# Adding connections
cond_herm_to_male_chem = sf.initialize_dimorphic_dict(names_dict_male_chem, len(Dimorphic_connections_Job), 1)
cond_herm_to_male_gap = sf.initialize_dimorphic_dict(names_dict_male_gap, len(Dimorphic_connections_Job), 1)
cond_male_to_herm_chem = sf.initialize_dimorphic_dict(names_dict_herm_chem, len(Dimorphic_connections_Job), 1)
cond_male_to_herm_gap = sf.initialize_dimorphic_dict(names_dict_herm_gap, len(Dimorphic_connections_Job), 1)

# Removing connections
cond_herm_no_dimorphic_chem = sf.initialize_dimorphic_dict(names_dict_herm_chem, len(Dimorphic_connections_Job), 1)
cond_herm_no_dimorphic_gap = sf.initialize_dimorphic_dict(names_dict_herm_gap, len(Dimorphic_connections_Job), 1)
cond_male_no_dimorphic_chem = sf.initialize_dimorphic_dict(names_dict_male_chem, len(Dimorphic_connections_Job), 1)
cond_male_no_dimorphic_gap = sf.initialize_dimorphic_dict(names_dict_male_gap, len(Dimorphic_connections_Job), 1)

for row in range(Dimorphic_connections_Job.index[0],Dimorphic_connections_Job.index[len(Dimorphic_connections_Job)-1]+1):
    Sgap_herm = sf.str_to_nparray(Dimorphic_connections_Job.Sgap_herm_adj[row], (len(neurons), len(neurons))) 
    Schem_herm = sf.str_to_nparray(Dimorphic_connections_Job.Schem_herm_adj[row],  (len(neurons), len(neurons))) 
    Sgap_male = sf.str_to_nparray(Dimorphic_connections_Job.Sgap_male_adj[row],  (len(neurons), len(neurons))) 
    Schem_male = sf.str_to_nparray(Dimorphic_connections_Job.Schem_male_adj[row],  (len(neurons), len(neurons)))
    parameter_ind = Dimorphic_connections_Job.parameters_index[row] 
 
# =============================================================================
# Current calculations
# =============================================================================
    for voltage in range(len(sensory_voltage_changes)):
        sensory_input = sensory_input_calc(Parameters_df.Rin[parameter_ind], sensory_voltage_changes[voltage])
        I = sf.input_current(Sgap_herm, t, range(sensory_start, sensory_end), neurons.index("ASH"), sensory_input, 0, range(0,len(t)), neurons.index("AVA"), Parameters_df.I_AVA[parameter_ind], 0, range(0,len(t)), neurons.index("AVB"), Parameters_df.I_AVB[parameter_ind], 0, range(0,len(t)), neurons.index("PVC"), Parameters_df.I_PVC[parameter_ind], 0) # units: A

# =============================================================================
# Transferring the dimorphic connections to the opposite sex    
# =============================================================================

# =============================================================================
# Hermaphrodites to males adding connections
# =============================================================================
        # Chemical synapses
        herm_to_male_chem, cond_herm_to_male_chem = sf.single_con_sex_reversal_pol_samp(herm_to_male_chem, cond_herm_to_male_chem, male_ind_chem, male_cells_chem, Schem_herm, Schem_male, Sgap_herm, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "chem", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)

        # Gap juctions
        herm_to_male_gap, cond_herm_to_male_gap = sf.single_con_sex_reversal_pol_samp(herm_to_male_gap, cond_herm_to_male_gap, male_ind_gap, male_cells_gap, Sgap_herm, Sgap_male, Schem_herm, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "gap", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)

# =============================================================================
# Hermaphrodites removing dimorphic connections
# =============================================================================
# Here, the indices that needs to be copied from the other sex are the indices of the tested sex 
# (since we are removing dimorphic connections instead of adding missing connections)
        # Chemical synapses
        herm_no_dimorphic_chem, cond_herm_no_dimorphic_chem = sf.single_con_sex_reversal_pol_samp(herm_no_dimorphic_chem, cond_herm_no_dimorphic_chem, herm_ind_chem, herm_cells_chem, Schem_herm, Schem_male, Sgap_herm, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "chem", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)

        # Gap juctions
        herm_no_dimorphic_gap, cond_herm_no_dimorphic_gap = sf.single_con_sex_reversal_pol_samp(herm_no_dimorphic_gap, cond_herm_no_dimorphic_gap, herm_ind_gap, herm_cells_gap, Sgap_herm, Sgap_male, Schem_herm, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "gap", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)

# =============================================================================
# Males to hermaphrodites adding connections
# =============================================================================
        # Chemical synapses
        male_to_herm_chem, cond_male_to_herm_chem = sf.single_con_sex_reversal_pol_samp(male_to_herm_chem, cond_male_to_herm_chem, herm_ind_chem, herm_cells_chem, Schem_male, Schem_herm, Sgap_male, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "chem", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)

        # Gap juctions
        male_to_herm_gap, cond_male_to_herm_gap = sf.single_con_sex_reversal_pol_samp(male_to_herm_gap, cond_male_to_herm_gap, herm_ind_gap, herm_cells_gap, Sgap_male, Sgap_herm, Schem_male, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "gap", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)

# =============================================================================
# Males removing dimorphic connections
# =============================================================================
        # Chemical synapses
        male_no_dimorphic_chem, cond_male_no_dimorphic_chem = sf.single_con_sex_reversal_pol_samp(male_no_dimorphic_chem, cond_male_no_dimorphic_chem, male_ind_chem, male_cells_chem, Schem_male, Schem_herm, Sgap_male, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "chem", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)

        # Gap juctions
        male_no_dimorphic_gap, cond_male_no_dimorphic_gap = sf.single_con_sex_reversal_pol_samp(male_no_dimorphic_gap, cond_male_no_dimorphic_gap, male_ind_gap, male_cells_gap, Sgap_male, Sgap_herm, Schem_male, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, "gap", t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms)
        print("finished row "+str(row)+", sensory input "+str(voltage))

# =============================================================================
# Hermaphrodites to males adding connections

for con in herm_to_male_chem.keys():
    Dimorphic_connections_Job["conditions_masc_herm_chem_"+ con] = cond_herm_to_male_chem[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_masc_herm_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_to_male_chem[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_masc_herm_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_to_male_chem[con][:,sens_input]> 0, 1, 0)

for con in herm_to_male_gap.keys():
    Dimorphic_connections_Job["conditions_masc_herm_gap_"+ con] = cond_herm_to_male_gap[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_masc_herm_gap_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_to_male_gap[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_masc_herm_gap_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_to_male_gap[con][:,sens_input]> 0, 1, 0)

# Hermaphrodites removing dimorphic connections
for con in herm_no_dimorphic_chem.keys():
    Dimorphic_connections_Job["conditions_no_dimorphic_herm_chem_"+ con] = cond_herm_no_dimorphic_chem[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_no_dimorphic_herm_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_no_dimorphic_chem[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_no_dimorphic_herm_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_no_dimorphic_chem[con][:,sens_input]> 0, 1, 0)

for con in herm_no_dimorphic_gap.keys():
    Dimorphic_connections_Job["conditions_no_dimorphic_herm_gap_"+ con] = cond_herm_no_dimorphic_gap[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_no_dimorphic_herm_gap_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_no_dimorphic_gap[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_no_dimorphic_herm_gap_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_no_dimorphic_gap[con][:,sens_input]> 0, 1, 0)

# Males to hermaphrodites adding connections
for con in male_to_herm_chem.keys():
    Dimorphic_connections_Job["conditions_fem_male_chem_"+ con] = cond_male_to_herm_chem[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_fem_male_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_to_herm_chem[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_fem_male_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_to_herm_chem[con][:,sens_input]> 0, 1, 0)

for con in male_to_herm_gap.keys():
    Dimorphic_connections_Job["conditions_fem_male_gap_"+ con] = cond_male_to_herm_gap[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_fem_male_gap_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_to_herm_gap[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_fem_male_gap_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_to_herm_gap[con][:,sens_input]> 0, 1, 0)

# Males removing dimorphic connections
for con in male_no_dimorphic_chem.keys():
    Dimorphic_connections_Job["conditions_no_dimorphic_male_chem_"+ con] = cond_male_no_dimorphic_chem[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_no_dimorphic_male_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_no_dimorphic_chem[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_no_dimorphic_male_chem_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_no_dimorphic_chem[con][:,sens_input]> 0, 1, 0)

for con in male_no_dimorphic_gap.keys():
    Dimorphic_connections_Job["conditions_no_dimorphic_male_gap_"+ con] = cond_male_no_dimorphic_gap[con]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_connections_Job["response_no_dimorphic_male_gap_response_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_no_dimorphic_gap[con][:,sens_input]
        Dimorphic_connections_Job["A_vs_B_no_dimorphic_male_gap_"+ con + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_no_dimorphic_gap[con][:,sens_input]> 0, 1, 0)

Dimorphic_connections_Job.to_csv("250422_dimorphic_connections_ID_"+str(job_ID)+".csv")
