import numpy as np
import supporting_functions as sf
import pandas as pd
import os

# =============================================================================
# Define for cluster use
# =============================================================================
job_ID = int(os.environ['LSB_JOBINDEX']) - 1
job_num = 2000

# =============================================================================
# Read parameters sets
# this file contains all sets that were appropriate to either of the sexes, 
# after the adjustments to the connections' strength. 
# =============================================================================
herm_cell_activity_basic_mutual = pd.read_csv("herm_cell_activity_basic_mutual.csv")
Parameters_df = pd.read_csv("Parameters_combinations_df.csv", index_col = 0)

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
# Input that would trigger a specific voltage change:
# =============================================================================
# V = IR, I = V/R
sensory_input_calc = lambda res, volt: volt/res
sensory_voltage_changes = [75 * 10 **-3, 65 * 10 **-3, 55 * 10 ** -3, 45 * 10 **-3, 35 * 10 **-3, 25 * 10 ** -3, 15 * 10 **-3, 5 * 10 ** -3, 1 * 10 **-3]

# =============================================================================
# Cluster adjustments
# =============================================================================
job_segments = np.round(np.linspace(0, len(herm_cell_activity_basic_mutual), job_num+1))
job_segments = job_segments.astype('int')
Different_stimulus_Job = herm_cell_activity_basic_mutual[job_segments[job_ID]:job_segments[job_ID+1]].copy()

# =============================================================================
# Initializing response arrays
# =============================================================================
herm_response = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan
male_response = np.empty((len(Different_stimulus_Job),len(sensory_voltage_changes))) * np.nan

# =============================================================================
# Simulating sensory stimuli in varying strength
# =============================================================================
for row in range(Different_stimulus_Job.index[0],Different_stimulus_Job.index[len(Different_stimulus_Job)-1]+1):
    parameter_ind = Different_stimulus_Job.parameters_index[row] 
    Sgap_herm = sf.str_to_nparray(Different_stimulus_Job.Sgap_herm_adj[row], (len(neurons), len(neurons))) 
    Schem_herm = sf.str_to_nparray(Different_stimulus_Job.Schem_herm_adj[row],  (len(neurons), len(neurons))) 
    Sgap_male = sf.str_to_nparray(Different_stimulus_Job.Sgap_male_adj[row],  (len(neurons), len(neurons))) 
    Schem_male = sf.str_to_nparray(Different_stimulus_Job.Schem_male_adj[row],  (len(neurons), len(neurons)))    
    for voltage in range(len(sensory_voltage_changes)):
        sensory_input = sensory_input_calc(Parameters_df.Rin[parameter_ind], sensory_voltage_changes[voltage])
        I = sf.input_current(Sgap_herm, t, range(sensory_start, sensory_end), neurons.index("ASH"), sensory_input, 0, range(0,len(t)), neurons.index("AVA"), Parameters_df.I_AVA[parameter_ind], 0, range(0,len(t)), neurons.index("AVB"), Parameters_df.I_AVB[parameter_ind], 0, range(0,len(t)), neurons.index("PVC"), Parameters_df.I_PVC[parameter_ind], 0) # units: A
# =============================================================================
# Hermaphrodites
# =============================================================================
        V_h = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_herm, Sgap_herm)
# since we expect oscillations we check the average of 1 second to determine which is higher, and not a single time step. 
        herm_response[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_h[neurons.index("A"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_h[neurons.index("B"), sensory_end - (1000 * correction_to_ms) : sensory_end])

# =============================================================================
# Males
# =============================================================================
        V_m = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_male, Sgap_male)
# since we expect oscillations we check the average of 1 second to determine which is higher, and not a single time step. 
        male_response[row - Different_stimulus_Job.index[0], voltage] = np.mean(V_m[neurons.index("A"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_m[neurons.index("B"), sensory_end - (1000 * correction_to_ms) : sensory_end])

        print("finished row "+str(row)+", sensory input "+str(voltage))

for sens_input in range(len(sensory_voltage_changes)):
    Different_stimulus_Job["herm_reponse_"+str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_response[:,sens_input]

for sens_input in range(len(sensory_voltage_changes)):
    Different_stimulus_Job["male_reponse_"+str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_response[:,sens_input]

for sens_input in range(len(sensory_voltage_changes)):
     Different_stimulus_Job["herm_A_vs_B_"+str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_response[:,sens_input]> 0, 1, 0)

for sens_input in range(len(sensory_voltage_changes)):
     Different_stimulus_Job["male_A_vs_B_"+str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_response[:,sens_input]> 0, 1, 0)

Different_stimulus_Job.to_csv("200422_different_stimuli_samp_pol_ID_"+str(job_ID)+".csv")
