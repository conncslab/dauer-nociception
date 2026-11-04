import pandas as pd
from os import path
from tqdm import tqdm
import numpy as np

# Inset here the path to the folder in which the outputs of "find_parameters_polarity_all_single_flips" exist. 
path_to_files_folder = "Insert path here"

# =============================================================================
# Read files from cluster
# =============================================================================
# parameters_single_flips = [] 
# for file in tqdm(range(0,16000)):
#     if path.exists(path_to_files_folder +r"\020622_parameters_polarity_single_flips_ID_"+str(file)+".csv"):
#         single_run_df = pd.read_csv(path_to_files_folder +r"\020622_parameters_polarity_single_flips_ID_"+str(file)+".csv", index_col=0)
#         single_run_df["parameters_index"] = single_run_df["parameters_index"].astype("int32")
#         parameters_single_flips.append(single_run_df) 
#     else:
#         print("020622_parameters_polarity_single_flips_ID_"+str(file)+".csv"+" was not found")
# 
# parameters_single_flips_df = pd.concat(parameters_single_flips , ignore_index=True)
# print(parameters_single_flips_df.info())
# parameters_single_flips_df.to_csv("basic_true_single_pol_flips.csv")
# herm_physiological_basic_behavioral_single_flips = parameters_single_flips_df[(parameters_single_flips_df["mutual_basic_herm_activity"] == "True") | (parameters_single_flips_df["mutual_basic_herm_activity"] == True)].reset_index()
# herm_physiological_basic_behavioral_single_flips.to_csv("herm_physiological_basic_behavioral_single_flips.csv")
# =============================================================================

parameters_single_flips_df = pd.read_csv("basic_true_single_pol_flips.csv")
herm_physiological_basic_behavioral_single_flips = pd.read_csv("herm_physiological_basic_behavioral_single_flips.csv")

valid_sets_per_single_flip = pd.DataFrame({"Inhibiotory synapse" : list(set(parameters_single_flips_df["inhibitory_synapse"]))})
Herm_basic_true = []
Herm_all_true = []
Male_basic_true = []
Male_all_true = []
Mutual_basic = []
Mutual_basic_herm_cell_activity = []
Mutual_all_true = []
for synapse in valid_sets_per_single_flip["Inhibiotory synapse"]:
    relevant_polarity_df = parameters_single_flips_df[parameters_single_flips_df["inhibitory_synapse"] == synapse]
    Herm_basic_true.append(np.sum((relevant_polarity_df["herm_basic_conditions"] == "True") | (relevant_polarity_df["herm_basic_conditions"] == True)))
    Herm_all_true.append(np.sum((relevant_polarity_df["True_herm"] == "True") | (relevant_polarity_df["True_herm"] == True)))
    Male_basic_true.append(np.sum((relevant_polarity_df["male_basic_conditions"] == "True") | (relevant_polarity_df["male_basic_conditions"] == True)))
    Male_all_true.append(np.sum((relevant_polarity_df["True_male"] == "True") | (relevant_polarity_df["True_male"] == True)))
    Mutual_basic.append(np.sum(((relevant_polarity_df["male_basic_conditions"] == "True") | (relevant_polarity_df["male_basic_conditions"] == True)) & ((relevant_polarity_df["herm_basic_conditions"] == "True") | (relevant_polarity_df["herm_basic_conditions"] == True))))
    Mutual_basic_herm_cell_activity.append(np.sum((relevant_polarity_df["mutual_basic_herm_activity"] == "True") | (relevant_polarity_df["mutual_basic_herm_activity"] == True)))
    Mutual_all_true.append(np.sum(((relevant_polarity_df["True_herm"] == "True") | (relevant_polarity_df["True_herm"] == True)) & ((relevant_polarity_df["True_male"] == "True") | (relevant_polarity_df["True_male"] == True))))
valid_sets_per_single_flip["Herm_basic_true"] = Herm_basic_true
valid_sets_per_single_flip["Herm_all_true"] = Herm_all_true
valid_sets_per_single_flip["Male_basic_true"] = Male_basic_true
valid_sets_per_single_flip["Male_all_true"] = Male_all_true
valid_sets_per_single_flip["Mutual_basic"] = Mutual_basic
valid_sets_per_single_flip["Mutual_basic_herm_cell_activity"] = Mutual_basic_herm_cell_activity
valid_sets_per_single_flip["Mutual_all_true"] = Mutual_all_true
valid_sets_per_single_flip.to_csv("valid_sets_per_single_flip.csv")

