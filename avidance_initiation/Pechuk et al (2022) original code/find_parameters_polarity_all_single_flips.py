import pandas as pd
import numpy as np
import extract_connectivity_df as ecd
import supporting_functions as sf
import os

# =============================================================================
# Define for cluster use
# =============================================================================
job_num = 16000
job_ID = int(os.environ['LSB_JOBINDEX']) - 1

# =============================================================================
# Download needed files
# =============================================================================
Parameters_df = pd.read_csv("Parameters_combinations_df.csv", index_col=0)
path_to_cook_file = "41586_2019_1352_MOESM9_ESM.xlsx"

herm_chem_full_connectivity = pd.read_excel(path_to_cook_file, sheet_name = "hermaphrodite chemical", header = None)
herm_gapj_full_connectivity = pd.read_excel(path_to_cook_file, sheet_name = "herm gap jn symmetric", header = None)
male_chem_full_connectivity = pd.read_excel(path_to_cook_file, sheet_name = "male chemical", header = None)
male_gapj_full_connectivity = pd.read_excel(path_to_cook_file, sheet_name = "male gap jn symmetric", header = None)

# =============================================================================
# Define neurons of interest:
# In circuit_neurons the keys are the name of the neural class, and the values are
# the regex expressions to find the cells in Cook's data.  
# =============================================================================
circuit_neurons = {'ASH':'ASH[RL]', 'AVA':'AVA[RL]', 'AVD':'AVD[RL]', 'AVB':'AVB[RL]', 'PVC':'PVC[RL]', 'A':'[DV]A[0-9]', 'B':'[DV]B[0-9]'}
neurons = list(circuit_neurons.keys())

Sgap_herm_head, Schem_herm_head, Sgap_male_head, Schem_male_head = ecd.final_connectivity_matrix(herm_chem_full_connectivity, herm_gapj_full_connectivity, male_chem_full_connectivity, male_gapj_full_connectivity, circuit_neurons)

# =============================================================================
# Find all chemical connections (inhibition options)
# =============================================================================
# Hermaphrodites
herm_ind = np.where(Schem_herm_head != 0)
herm_synapses = [(neurons[herm_ind[1][i]], neurons[herm_ind[0][i]]) for i in range(len(herm_ind[0]))]

# Males
male_ind = np.where(Schem_male_head != 0)
male_synapses = [(neurons[male_ind[1][i]], neurons[male_ind[0][i]]) for i in range(len(male_ind[0]))]

union_synapses = list(set(herm_synapses) | set(male_synapses))
already_tested = [("ASH", "AVB"), ("AVB", "AVA"), ("A","B"), ("B","A")]
final_synapses = list(set(union_synapses) - set(already_tested))

# tested - all identified synapses match Figure 2A
# =============================================================================
# Find all combinations of polarity and parameters
# =============================================================================
parameters_pol_df = pd.DataFrame()
parameters_num = len(Parameters_df)
polarity_num = len(final_synapses)
parameters_pol_df["parameters_index"] = Parameters_df.index.tolist()
parameters_pol_df = pd.concat([parameters_pol_df] * polarity_num, ignore_index=True)
parameters_pol_df["inhibitory_synapse"] = sorted(final_synapses * parameters_num)
parameters_pol_df["parameters_index"] = parameters_pol_df["parameters_index"].astype("int32")

# =============================================================================
# setting the enviroment
# =============================================================================
# time:
t = np.linspace(0,15,75001) # 15 seconds, in 0.0002 (0.2ms) jumps
correction_to_ms = int(round(0.001/(t[1]-t[0])))
sensory_start = 5000 * correction_to_ms
sensory_end = 10000 * correction_to_ms

# resting state: 
Vrest = np.reshape(np.repeat(np.array([-0.04]),len(neurons)), (len(neurons),1)) #units: V

# =============================================================================
# Cluster adjustments
# =============================================================================
job_segments = np.round(np.linspace(0, len(parameters_pol_df), job_num+1))
job_segments = job_segments.astype('int')
Parameters_Job = parameters_pol_df[job_segments[job_ID]:job_segments[job_ID+1]].copy()

# =============================================================================
# initializing the conditions
# =============================================================================
herm_basic_conditions = []
herm_cell_act_conditions = []
male_basic_conditions = []
male_cell_act_conditions = []
time_constant = []
Sgap_herm_adj = []
Schem_herm_adj = []
Sgap_male_adj = []
Schem_male_adj = []

for row in range(Parameters_Job.index[0],Parameters_Job.index[len(Parameters_Job)-1]+1):
    parameter_ind = Parameters_Job.parameters_index[row] 
    polarity = Parameters_Job.inhibitory_synapse[row]
# =============================================================================
# Randomly adjusting the strength of the connections
# =============================================================================
    Schem_herm_row, mult_chem_herm = ecd.adjusting_strengths(Schem_herm_head,[1.2, 1, 0.8], "chem")
    Sgap_herm_row, mult_gap_herm =  ecd.adjusting_strengths(Sgap_herm_head,[1.2, 1, 0.8], "gap")
    Schem_male_row, mult_chem_male =  ecd.adjusting_strengths(Schem_male_head,[1.2, 1, 0.8], "chem")
    Sgap_male_row, mult_gap_male =  ecd.adjusting_strengths(Sgap_male_head,[1.2, 1, 0.8], "gap")

# =============================================================================
# Update polarity
# =============================================================================
    Schem_herm_row = sf.single_flip_change(Schem_herm_row, neurons, polarity)
    Schem_male_row = sf.single_flip_change(Schem_male_row, neurons, polarity)

    Sgap_herm_adj.append(Sgap_herm_row)
    Schem_herm_adj.append(Schem_herm_row)
    Sgap_male_adj.append(Sgap_male_row)
    Schem_male_adj.append(Schem_male_row)

# =============================================================================
# Define input current:
# =============================================================================
    I = sf.input_current(Sgap_herm_row, t, range(sensory_start, sensory_end), neurons.index("ASH"), Parameters_df.Sensory_Input[parameter_ind], 0, range(0,len(t)), neurons.index("AVA"), Parameters_df.I_AVA[parameter_ind], 0, range(0,len(t)), neurons.index("AVB"), Parameters_df.I_AVB[parameter_ind], 0, range(0,len(t)),  neurons.index("PVC"), Parameters_df.I_PVC[parameter_ind], 0) # units: A

# =============================================================================
# Hermaphrodites
# =============================================================================
    sex = "Hermaphrodites"
    V_h_adj_con = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_herm_row, Sgap_herm_row, sex)
    basic_conditions_herm = sf.basic_conditions_check(V_h_adj_con, t, neurons, sensory_start, sensory_end, correction_to_ms)
    herm_basic_conditions.append(basic_conditions_herm)
    if basic_conditions_herm == (True):
        herm_cell_act_conditions.append(sf.cells_activity_conditions_check(V_h_adj_con, neurons, sensory_start, sensory_end, correction_to_ms))
    else:
        herm_cell_act_conditions.append(("irrelevant"))
# =============================================================================
# Males
# =============================================================================
    sex = "Males"
    V_m_adj_con = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_male_row, Sgap_male_row, sex)
    basic_conditions_male = sf.basic_conditions_check(V_m_adj_con, t, neurons, sensory_start, sensory_end, correction_to_ms)
    male_basic_conditions.append(basic_conditions_male)
    if basic_conditions_male == (True):
        male_cell_act_conditions.append(sf.cells_activity_conditions_check(V_m_adj_con, neurons,sensory_start, sensory_end, correction_to_ms))
    else:
        male_cell_act_conditions.append(("irrelevant"))

# =============================================================================
# General
# =============================================================================
    time_constant.append(Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind] < 500 * 10 **-3)
    print("finished row", row)

# =============================================================================
# Save to dataframe
# =============================================================================
Parameters_Job["herm_basic_conditions"] = herm_basic_conditions
Parameters_Job["male_basic_conditions"] = male_basic_conditions
Parameters_Job["herm_cell_act_conditions"] = herm_cell_act_conditions
Parameters_Job["male_cell_act_conditions"] = male_cell_act_conditions
Parameters_Job["time_constant"] = time_constant
Parameters_Job["Schem_herm_adj"] = Schem_herm_adj
Parameters_Job["Sgap_herm_adj"] = Sgap_herm_adj
Parameters_Job["Schem_male_adj"] = Schem_male_adj
Parameters_Job["Sgap_male_adj"] = Sgap_male_adj
Parameters_Job["True_herm"] = ((Parameters_Job["herm_basic_conditions"] == True) | (Parameters_Job["herm_basic_conditions"] == "True")) & ((Parameters_Job["herm_cell_act_conditions"] == True) | (Parameters_Job["herm_cell_act_conditions"] == "True"))  
Parameters_Job["True_male"] = ((Parameters_Job["male_basic_conditions"] == True) | (Parameters_Job["male_basic_conditions"] == "True")) & ((Parameters_Job["male_cell_act_conditions"] == True) | (Parameters_Job["male_cell_act_conditions"] == "True"))  
Parameters_Job["mutual_basic_herm_activity"] = ((Parameters_Job["herm_basic_conditions"] == True) | (Parameters_Job["herm_basic_conditions"] == "True")) & ((Parameters_Job["herm_cell_act_conditions"] == True) | (Parameters_Job["herm_cell_act_conditions"] == "True")) & ((Parameters_Job["male_basic_conditions"] == True) | (Parameters_Job["male_basic_conditions"] == "True")) 

some_true =  Parameters_Job[((Parameters_Job["herm_basic_conditions"] == "True") | (Parameters_Job["herm_basic_conditions"] == True) | (Parameters_Job["male_basic_conditions"] == "True") | (Parameters_Job["male_basic_conditions"] == True)) & (Parameters_Job["time_constant"] == True)]
some_true.to_csv("020622_parameters_polarity_single_flips_ID_"+str(job_ID)+".csv")
