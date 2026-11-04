import numpy as np
import supporting_functions as sf
import pandas as pd
import os

# =============================================================================
# Define for cluster use
# =============================================================================
job_ID = int(os.environ['LSB_JOBINDEX']) - 1
job_num = 200

# =============================================================================
# Read parameters sets
# this file contains all sets that were appropriate for both sexes, 
# after the adjustments to the connections' strength. 
# =============================================================================
mutual_sets = pd.read_csv("herm_cell_activity_basic_mutual.csv")
mutual_sets_final = mutual_sets[mutual_sets["polarity_index"] == 8].reset_index()
Parameters_df = pd.read_csv("Parameters_combinations_df.csv", index_col = 0)

# =============================================================================
# Set environment
# =============================================================================

# define neurons of interest:
circuit_neurons = {'ASH':'ASH[RL]', 'AVA':'AVA[RL]', 'AVD':'AVD[RL]', 'AVB':'AVB[RL]', 'PVC':'PVC[RL]', 'A':'[DV]A[0-9]', 'B':'[DV]B[0-9]'}
neurons = list(circuit_neurons.keys())
tested_interneurons = ["AVA", "AVD", "AVB", "PVC"]

# time:
t = np.linspace(0,15,75001) # 15 seconds, in 0.0002 (0.2ms) jumps
correction_to_ms = int(round(0.001/(t[1]-t[0])))
sensory_start = 5000 * correction_to_ms
sensory_end = 10000 * correction_to_ms

# resting state: 
Vrest = np.reshape(np.repeat(np.array([-0.04]),len(neurons)), (len(neurons),1)) #units: V

# =============================================================================
# Input that would trigger a specific voltage change:
# =============================================================================
# V = IR, I = V/R
sensory_input_calc = lambda res, volt: volt/res
sensory_voltage_changes = [75 * 10 **-3, 45 * 10 ** -3, 15 * 10 **-3, 1 * 10 **-3]

# =============================================================================
# Cluster adjustments
# =============================================================================
job_segments = np.round(np.linspace(0, len(mutual_sets_final), job_num+1))
job_segments = job_segments.astype('int')
Different_stimulus_Job = mutual_sets_final[job_segments[job_ID]:job_segments[job_ID+1]].copy()

# =============================================================================
# Initializing response arrays
# =============================================================================
herm_AVA = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan
herm_AVD = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan
herm_AVB = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan
herm_PVC = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan

male_AVA = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan
male_AVD = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan
male_AVB = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan
male_PVC = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan

# =============================================================================
# Simulating sensory stimuli in varying strength and finding the activations of the interneurons
# =============================================================================
for row in range(Different_stimulus_Job.index[0],Different_stimulus_Job.index[len(Different_stimulus_Job)-1]+1):
    Sgap_herm = sf.str_to_nparray(Different_stimulus_Job.Sgap_herm_adj[row], (len(neurons), len(neurons))) 
    Schem_herm = sf.str_to_nparray(Different_stimulus_Job.Schem_herm_adj[row],  (len(neurons), len(neurons))) 
    Sgap_male = sf.str_to_nparray(Different_stimulus_Job.Sgap_male_adj[row],  (len(neurons), len(neurons))) 
    Schem_male = sf.str_to_nparray(Different_stimulus_Job.Schem_male_adj[row],  (len(neurons), len(neurons)))    
    parameter_ind = Different_stimulus_Job.parameters_index[row] 
  
    for voltage in range(len(sensory_voltage_changes)):
        sensory_input = sensory_input_calc(Parameters_df.Rin[parameter_ind], sensory_voltage_changes[voltage])
        I = sf.input_current(Sgap_herm, t, range(sensory_start, sensory_end), neurons.index("ASH"), sensory_input, 0, range(0,len(t)), neurons.index("AVA"), Parameters_df.I_AVA[parameter_ind], 0, range(0,len(t)), neurons.index("AVB"), Parameters_df.I_AVB[parameter_ind], 0, range(0,len(t)), neurons.index("PVC"), Parameters_df.I_PVC[parameter_ind], 0) # units: A
        # Hermaphrodite
        V_h = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_herm, Sgap_herm)
        herm_AVA[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_h[neurons.index("AVA"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_h[neurons.index("AVA"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        herm_AVD[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_h[neurons.index("AVD"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_h[neurons.index("AVD"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        herm_AVB[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_h[neurons.index("AVB"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_h[neurons.index("AVB"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        herm_PVC[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_h[neurons.index("PVC"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_h[neurons.index("PVC"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        
        # Male
        V_m = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_male, Sgap_male)
        male_AVA[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_m[neurons.index("AVA"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_m[neurons.index("AVA"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        male_AVD[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_m[neurons.index("AVD"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_m[neurons.index("AVD"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        male_AVB[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_m[neurons.index("AVB"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_m[neurons.index("AVB"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        male_PVC[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_m[neurons.index("PVC"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_m[neurons.index("PVC"), sensory_start - (1000 * correction_to_ms) : sensory_start])
        print("finished row "+str(row)+", sensory input "+str(voltage))

for sens_input in range(len(sensory_voltage_changes)):
    Different_stimulus_Job["herm_AVA_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_AVA[:,sens_input]
    Different_stimulus_Job["herm_AVD_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_AVD[:,sens_input]
    Different_stimulus_Job["herm_AVB_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_AVB[:,sens_input]
    Different_stimulus_Job["herm_PVC_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_PVC[:,sens_input]
    
    Different_stimulus_Job["male_AVA_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_AVA[:,sens_input]
    Different_stimulus_Job["male_AVD_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_AVD[:,sens_input]
    Different_stimulus_Job["male_AVB_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_AVB[:,sens_input]
    Different_stimulus_Job["male_PVC_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_PVC[:,sens_input]

Different_stimulus_Job.to_csv("280422_interneurons_activations_ID_"+str(job_ID)+".csv")

