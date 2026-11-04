import pandas as pd
import numpy as np
from itertools import product

# =============================================================================
# Parameters' combinations
# From each parameter 7 values were sampled, equally distributed on a logarithmic scale.
# =============================================================================
resistance_per_area = np.geomspace(1500 * 10 **3, 15 * 10**3, 7)
capacitance = np.geomspace(10 **-7 , 10**-5, 7)
gchem = np.geomspace(10**-9, 10**-12, 7)
ggap = np.geomspace(10**-9, 10**-12, 7)
AVB_dV = np.geomspace(75 * 10**-3,  1* 10**-3, 7)
AVA_dV = np.geomspace(75 * 10**-3,  1* 10**-3, 7)
PVC_dV = np.geomspace(75 * 10**-3,  1* 10**-3, 7)
[RES, CAP, GCHEM, GGAP, AVB_DV, AVA_DV, PVC_DV]= np.meshgrid(resistance_per_area, capacitance, gchem,ggap, AVB_dV, AVA_dV, PVC_dV)
len_df = len(resistance_per_area) * len(capacitance) * len(gchem) * len(ggap) *len(AVB_dV) *len(AVA_dV) *len(PVC_dV)
dict_for_df = {"Rm": np.reshape(RES, len_df), "Capacitance" : np.reshape(CAP, len_df),
               "Chemical_Conductance" : np.reshape(GCHEM, len_df), "Gap_Conductance" : np.reshape(GGAP, len_df),
               "Voltage_Change_AVB": np.reshape(AVB_DV, len_df),"Voltage_Change_AVA": np.reshape(AVA_DV, len_df),
               "Voltage_Change_PVC": np.reshape(PVC_DV, len_df)}
Parameters_df = pd.DataFrame(dict_for_df)

# fixed parameters:
surace_area = 15 *10**-6 # units: cm2
Rin = Parameters_df.Rm.map(lambda x: x/surace_area)
Parameters_df["Rin"] = Rin

# input that would trigger 100mV change:
# V = IR
# 75 * 10**-3 = I * Rin
sensory_I = Parameters_df.Rin.map(lambda x: 75 * 10**-3/x)
Parameters_df["Sensory_Input"] = sensory_I

# input to the interneurons: 
I_AVB = Parameters_df.Voltage_Change_AVB/Parameters_df.Rin
I_AVA = Parameters_df.Voltage_Change_AVA/Parameters_df.Rin
I_PVC = Parameters_df.Voltage_Change_PVC/Parameters_df.Rin

Parameters_df["I_AVB"] = I_AVB
Parameters_df["I_AVA"] = I_AVA
Parameters_df["I_PVC"] = I_PVC

Parameters_df.to_csv("Parameters_combinations_df.csv")
# =============================================================================
# Create polarity combinations
# =============================================================================
polarity_options = [("ASH", "AVB"), ("AVB", "AVA"), ("A","B"), ("B","A")]
polarity_mult = [1,-1]
possible_combinations = list(product(polarity_mult, repeat = len(polarity_options)))
possible_combinations_df = pd.DataFrame(possible_combinations, columns = polarity_options)
num_combs = len(possible_combinations_df)
possible_combinations_df.to_pickle("Polarity_combinations_sample.pkl")

