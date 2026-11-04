import pandas as pd
from os import path
from tqdm import tqdm
import numpy as np
import random
import supporting_functions as sf

# Inset here the path to the folder in which the outputs of "find_prarmeters_polarity_4_connections" exist. 
path_to_files_folder = "Insert path here"

# =============================================================================
# parameters_pol_samp = [] 
# for file in tqdm(range(0,16000)):
#     if path.exists(path_to_files_folder +r"\1804_parameters_polarity_sample_ID_"+str(file)+".csv"):
#         single_run_df = pd.read_csv(path_to_files_folder +r"\1804_parameters_polarity_sample_ID_"+str(file)+".csv", index_col=0)
#         single_run_df["parameters_index"] = single_run_df["parameters_index"].astype("int32")
#         single_run_df["polarity_index"] = single_run_df["polarity_index"].astype("int8")
#         some_true =  single_run_df[((single_run_df["herm_basic_conditions"] == "True") | (single_run_df["herm_basic_conditions"] == True) | (single_run_df["male_basic_conditions"] == "True") | (single_run_df["male_basic_conditions"] == True)) & (single_run_df["time_constant"] == True)]
#         parameters_pol_samp.append(some_true) 
#     else:
#         print("1804_parameters_polarity_sample_ID_"+str(file)+".csv"+" was not found")
# 
# parameters_pol_samp_df = pd.concat(parameters_pol_samp , ignore_index=True)
# print(parameters_pol_samp_df.info())
# parameters_pol_samp_df.to_csv("basic_true_par_pol_samp.csv")
# herm_cell_activity_basic_mutual = parameters_pol_samp_df[(parameters_pol_samp_df["mutual_basic_herm_activity"] == "True") | (parameters_pol_samp_df["mutual_basic_herm_activity"] == True)].reset_index()
# herm_cell_activity_basic_mutual.to_csv("herm_cell_activity_basic_mutual.csv")
# =============================================================================

parameters_pol_samp_df = pd.read_csv("basic_true_par_pol_samp.csv")
herm_cell_activity_basic_mutual = pd.read_csv("herm_cell_activity_basic_mutual.csv")

possible_combinations_df = pd.read_pickle("Polarity_combinations_sample.pkl")
valid_sets_per_polarity = possible_combinations_df.copy()
Herm_basic_true = []
Herm_all_true = []
Male_basic_true = []
Male_all_true = []
Mutual_basic = []
Mutual_basic_herm_cell_activity = []
Mutual_all_true = []
for pol_ind in range(len(valid_sets_per_polarity)):
    relevant_polarity_df = parameters_pol_samp_df[parameters_pol_samp_df["polarity_index"] == pol_ind]
    Herm_basic_true.append(np.sum((relevant_polarity_df["herm_basic_conditions"] == "True") | (relevant_polarity_df["herm_basic_conditions"] == True)))
    Herm_all_true.append(np.sum((relevant_polarity_df["True_herm"] == "True") | (relevant_polarity_df["True_herm"] == True)))
    Male_basic_true.append(np.sum((relevant_polarity_df["male_basic_conditions"] == "True") | (relevant_polarity_df["male_basic_conditions"] == True)))
    Male_all_true.append(np.sum((relevant_polarity_df["True_male"] == "True") | (relevant_polarity_df["True_male"] == True)))
    Mutual_basic.append(np.sum(((relevant_polarity_df["male_basic_conditions"] == "True") | (relevant_polarity_df["male_basic_conditions"] == True)) & ((relevant_polarity_df["herm_basic_conditions"] == "True") | (relevant_polarity_df["herm_basic_conditions"] == True))))
    Mutual_basic_herm_cell_activity.append(np.sum((relevant_polarity_df["mutual_basic_herm_activity"] == "True") | (relevant_polarity_df["mutual_basic_herm_activity"] == True)))
    Mutual_all_true.append(np.sum(((relevant_polarity_df["True_herm"] == "True") | (relevant_polarity_df["True_herm"] == True)) & ((relevant_polarity_df["True_male"] == "True") | (relevant_polarity_df["True_male"] == True))))
valid_sets_per_polarity["Herm_basic_true"] = Herm_basic_true
valid_sets_per_polarity["Herm_all_true"] = Herm_all_true
valid_sets_per_polarity["Male_basic_true"] = Male_basic_true
valid_sets_per_polarity["Male_all_true"] = Male_all_true
valid_sets_per_polarity["Mutual_basic"] = Mutual_basic
valid_sets_per_polarity["Mutual_basic_herm_cell_activity"] = Mutual_basic_herm_cell_activity
valid_sets_per_polarity["Mutual_all_true"] = Mutual_all_true
valid_sets_per_polarity.to_csv("valid_sets_per_polarity.csv")

# =============================================================================
# simulate single set
# =============================================================================
df = herm_cell_activity_basic_mutual[herm_cell_activity_basic_mutual["polarity_index"] == 8].copy()
Parameters_df = pd.read_csv("Parameters_combinations_df.csv", index_col=0)

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

# row = random.choice(df.index)
row = 220 # This row was randomly chosen and saved for reproducibility of graphs
print("row: ", row)

Sgap_herm = sf.str_to_nparray(df.Sgap_herm_adj[row], (len(neurons), len(neurons))) 
Schem_herm = sf.str_to_nparray(df.Schem_herm_adj[row],  (len(neurons), len(neurons))) 
Sgap_male = sf.str_to_nparray(df.Sgap_male_adj[row],  (len(neurons), len(neurons))) 
Schem_male = sf.str_to_nparray(df.Schem_male_adj[row],  (len(neurons), len(neurons)))    
parameter_ind = df.parameters_index[row] 

I = sf.input_current(Sgap_herm, t, range(sensory_start, sensory_end), neurons.index("ASH"), Parameters_df.Sensory_Input[parameter_ind], 0, range(0,len(t)), neurons.index("AVA"), Parameters_df.I_AVA[parameter_ind], 0, range(0,len(t)), neurons.index("AVB"), Parameters_df.I_AVB[parameter_ind], 0, range(0,len(t)),  neurons.index("PVC"), Parameters_df.I_PVC[parameter_ind], 0) # units: A
V_h = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_herm, Sgap_herm, "herm")
V_m = sf.neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_male, Sgap_male, "male")

path_for_graphs_ss = r"saved_graphs\\single_sets"
sf.joint_plot_broken_y_axis(t, V_h, neurons, (27, 42), (-42, -27), path_for_graphs_ss, "Hermaphrodite")
sf.joint_plot_broken_y_axis(t, V_m, neurons, (27, 42), (-42, -27), path_for_graphs_ss, "Male")

print("Row details:\n", Parameters_df.iloc[parameter_ind])
print("Row polarity:\n", df["polarity_index"].iloc[row])

parameters = {'Rm':[r'(k' +'\u03A9' +r'$\cdot$' +r'$cm^2$' +r')', 1000], 'Capacitance':[r'(uF/' +r'$cm^2$' + r')', 10**-6], 'Chemical_Conductance':[r'(nS)', 10**-9], 'Gap_Conductance':[r'(nS)', 10**-9],'Voltage_Change_AVA':[r'(mV)', 10**-3], 'Voltage_Change_AVB':[r'(mV)', 10**-3], 'Voltage_Change_PVC':[r'(mV)', 10**-3]}
path_for_graphs_ph = r"saved_graphs\\parameters_histograms"
for pol_ind in range(8,16):
    rlevant_df = herm_cell_activity_basic_mutual[herm_cell_activity_basic_mutual["polarity_index"] == pol_ind]
    sf.bar_plot_per_parameter(rlevant_df, Parameters_df, parameters, path_for_graphs_ph)
