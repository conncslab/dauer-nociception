import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import supporting_functions as sf

# Inset here the path to the folder in which the outputs of "find_different_sensory_stimuli_4_connections" exist. 
path_to_files_folder = "Insert path here"

# =============================================================================
# Read files from cluster and save to csv
# =============================================================================
# different_stimuli_samp_pol_list = []
# for file in tqdm(range(2000)): 
#     if path.exists(path_to_files_folder + r"\200422_different_stimuli_samp_pol_ID_"+str(file)+".csv"):
#         single_run_df = pd.read_csv(path_to_files_folder + r"\200422_different_stimuli_samp_pol_ID_"+str(file)+".csv")
#         different_stimuli_samp_pol_list.append(single_run_df)
#     else:
#         print("200422_different_stimuli_samp_pol_ID_"+str(file)+".csv"+" was not found")
# 
# different_stimuli_samp_pol = pd.concat(different_stimuli_samp_pol_list , ignore_index=True)
# print(different_stimuli_samp_pol.info())
# 
# different_stimuli_samp_pol.to_csv("different_stimuli_samp_pol.csv")
# =============================================================================

different_stimuli_samp_pol = pd.read_csv("different_stimuli_samp_pol.csv")
title_1 = r"saved_graphs\\different_sensory_stimuli"

# Here we add 1 to the polarity index such that the first index would be 1 instead of 0. this is the only place in which it happens. 
sensory_voltage_changes = [1.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
for pol_ind in range(16):
    if len(different_stimuli_samp_pol[different_stimuli_samp_pol["polarity_index"] == pol_ind])>0:
        sf.plot_reverse_probability_per_input(sensory_voltage_changes, different_stimuli_samp_pol[different_stimuli_samp_pol["polarity_index"] == pol_ind], title_1 + r"\\Polarity index = " + str(pol_ind + 1), (0.3,-0.45))

title_2 = r"saved_graphs\\overlapping_graphs\\Aversive behavior all polarity combinations"
sf.plot_stacked_reverse_probability(sensory_voltage_changes, [different_stimuli_samp_pol[different_stimuli_samp_pol["polarity_index"] == pol_ind] for pol_ind in list(set(different_stimuli_samp_pol["polarity_index"]))], ["herm_A_vs_B_", "male_A_vs_B_"], ["Wild-type hermaphrodites","Wild-type males"], ["sandybrown", "darkturquoise"], title_2, (0.3,-0.45))
