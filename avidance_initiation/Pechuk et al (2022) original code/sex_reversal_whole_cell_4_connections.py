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
Dimorphic_cells_Job = sensory_strength_head[job_segments[job_ID]:job_segments[job_ID+1]].copy()

# =============================================================================
# Initializing response arrays
# =============================================================================
names_dict_herm_to_male = ["masculinized_" + neuron for neuron in neurons]
names_dict_male_to_herm = ["feminized_" + neuron for neuron in neurons]

# Responses
herm_to_male = sf.initialize_dimorphic_dict(names_dict_herm_to_male, len(Dimorphic_cells_Job), len(sensory_voltage_changes))
male_to_herm = sf.initialize_dimorphic_dict(names_dict_male_to_herm, len(Dimorphic_cells_Job), len(sensory_voltage_changes))

# Checking conditions
cond_herm_to_male = sf.initialize_dimorphic_dict(names_dict_herm_to_male, len(Dimorphic_cells_Job), 1)
cond_male_to_herm = sf.initialize_dimorphic_dict(names_dict_male_to_herm, len(Dimorphic_cells_Job), 1)

for row in range(Dimorphic_cells_Job.index[0],Dimorphic_cells_Job.index[len(Dimorphic_cells_Job)-1]+1):
    Sgap_herm = sf.str_to_nparray(Dimorphic_cells_Job.Sgap_herm_adj[row], (len(neurons), len(neurons))) 
    Schem_herm = sf.str_to_nparray(Dimorphic_cells_Job.Schem_herm_adj[row],  (len(neurons), len(neurons))) 
    Sgap_male = sf.str_to_nparray(Dimorphic_cells_Job.Sgap_male_adj[row],  (len(neurons), len(neurons))) 
    Schem_male = sf.str_to_nparray(Dimorphic_cells_Job.Schem_male_adj[row],  (len(neurons), len(neurons)))
    parameter_ind = Dimorphic_cells_Job.parameters_index[row]
    
# =============================================================================
# Current calculations
# =============================================================================
    for voltage in range(len(sensory_voltage_changes)):
        sensory_input = sensory_input_calc(Parameters_df.Rin[parameter_ind], sensory_voltage_changes[voltage])
        I = sf.input_current(Sgap_herm, t, range(sensory_start, sensory_end), neurons.index("ASH"), sensory_input, 0, range(0,len(t)), neurons.index("AVA"), Parameters_df.I_AVA[parameter_ind], 0, range(0,len(t)), neurons.index("AVB"), Parameters_df.I_AVB[parameter_ind], 0, range(0,len(t)), neurons.index("PVC"), Parameters_df.I_PVC[parameter_ind], 0) # units: A

# =============================================================================
# Transferring cell's connections to the opposite sex    
# =============================================================================
# Hermaphrodites to males
        for cell_ind in range(len(neurons)):
            Schem_herm_masculinized = Schem_herm.copy()
            Schem_herm_masculinized[cell_ind,:] = Schem_male[cell_ind,:] #out connections
            Schem_herm_masculinized[:,cell_ind] = Schem_male[:,cell_ind] #in connection
            Sgap_herm_masculinized = Sgap_herm.copy()
            Sgap_herm_masculinized[cell_ind,:] = Sgap_male[cell_ind,:] #out connections
            Sgap_herm_masculinized[:,cell_ind] = Sgap_male[:,cell_ind] #in connection
            
            V_h_masc = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_herm_masculinized, Sgap_herm_masculinized)
            herm_to_male["masculinized_"+ neurons[cell_ind]][row - Dimorphic_cells_Job.index[0], voltage] = np.mean(V_h_masc[neurons.index("A"), sensory_end - (1000 * correction_to_ms): sensory_end]) - np.mean(V_h_masc[neurons.index("B"), sensory_end - (1000 * correction_to_ms): sensory_end])
            if sensory_voltage_changes[voltage] == 75 * 10 **-3:
               cond_herm_to_male["masculinized_"+ neurons[cell_ind]][row - Dimorphic_cells_Job.index[0]] = sf.basic_conditions_check(V_h_masc, t, neurons, sensory_start, sensory_end, correction_to_ms)

# Males to hermaphrodites
            Schem_male_feminized = Schem_male.copy()
            Schem_male_feminized[cell_ind,:] = Schem_herm[cell_ind,:] #out connections
            Schem_male_feminized[:,cell_ind] = Schem_herm[:,cell_ind] #in connection
            Sgap_male_feminized = Sgap_male.copy()
            Sgap_male_feminized[cell_ind,:] = Sgap_herm[cell_ind,:] #out connections
            Sgap_male_feminized[:,cell_ind] = Sgap_herm[:,cell_ind] #in connection
            
            V_m_fem = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_male_feminized, Sgap_male_feminized)
            male_to_herm["feminized_"+ neurons[cell_ind]][row - Dimorphic_cells_Job.index[0], voltage] = np.mean(V_m_fem[neurons.index("A"), sensory_end - (1000 * correction_to_ms): sensory_end]) - np.mean(V_m_fem[neurons.index("B"), sensory_end - (1000 * correction_to_ms): sensory_end])
            if sensory_voltage_changes[voltage] == 75 * 10 **-3:
               cond_male_to_herm["feminized_"+ neurons[cell_ind]][row - Dimorphic_cells_Job.index[0]] = sf.basic_conditions_check(V_m_fem, t, neurons, sensory_start, sensory_end, correction_to_ms)

        print("finished row "+str(row)+", sensory input "+str(voltage))

# =============================================================================
# Hermaphrodites to males - update the dataframe
# =============================================================================
for cell1 in herm_to_male.keys():
    Dimorphic_cells_Job["conditions_masc_herm_"+ cell1] = cond_herm_to_male[cell1]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_cells_Job["response_masc_herm_"+ cell1 + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = herm_to_male[cell1][:,sens_input]
        Dimorphic_cells_Job["A_vs_B_masc_herm_"+ cell1 + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(herm_to_male[cell1][:,sens_input]> 0, 1, 0)

# =============================================================================
# males to hermaphrodites - update the dataframe
# =============================================================================
for cell2 in male_to_herm.keys():
    Dimorphic_cells_Job["conditions_fem_male_"+ cell2] = cond_male_to_herm[cell2]
    for sens_input in range(len(sensory_voltage_changes)):
        Dimorphic_cells_Job["response_fem_male_"+ cell2 + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = male_to_herm[cell2][:,sens_input]
        Dimorphic_cells_Job["A_vs_B_fem_male_"+ cell2 + "_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] = np.where(male_to_herm[cell2][:,sens_input]> 0, 1, 0)

Dimorphic_cells_Job.to_csv("250422_dimorphic_cells_ID_"+str(job_ID)+".csv")
    