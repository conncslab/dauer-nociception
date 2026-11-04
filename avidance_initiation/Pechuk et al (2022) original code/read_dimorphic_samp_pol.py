import pandas as pd
import supporting_functions as sf
from os import path
from tqdm import tqdm
# =============================================================================
# read files from cluster and save to csv
# =============================================================================
# =============================================================================
# Inset here the path to the folder in which the outputs of "sex_reversal_single_con_4_connections" exist. 
path_to_files_folder_single = "Insert path here"

# # Single connection
# Sex_reverse_single_con_list = []
# for file in tqdm(range(0,2000)):
#     if path.exists(path_to_files_folder_single + r"\250422_dimorphic_connections_ID_"+str(file)+".csv"):
#         single_run_df1 = pd.read_csv(path_to_files_folder_single + r"\250422_dimorphic_connections_ID_"+str(file)+".csv")
#         Sex_reverse_single_con_list.append(single_run_df1)    
#     else:
#         print("250422_dimorphic_connections_ID_"+str(file)+".csv"+" was not found")
# 
# Sex_reverse_single_con = pd.concat(Sex_reverse_single_con_list , ignore_index=True)
# Sex_reverse_single_con.to_csv("Sex_reverse_single_con_pol_samp_df.csv")
# 
# Inset here the path to the folder in which the outputs of "sex_reversal_double_con_4_connections" exist. 
path_to_files_folder_double = "Insert path here"
# # double connection
# Sex_reverse_double_con_list = []
# for file in tqdm(range(0,2000)):
#     if path.exists(path_to_files_folder_double + r"\250422_dimorphic_double_con_ID_"+str(file)+".csv"):
#         single_run_df2 = pd.read_csv(path_to_files_folder_double + r"\250422_dimorphic_double_con_ID_"+str(file)+".csv")
#         Sex_reverse_double_con_list.append(single_run_df2)    
#     else:
#         print("250422_dimorphic_double_con_ID_"+str(file)+".csv"+" was not found")
# 
# Sex_reverse_double_con = pd.concat(Sex_reverse_double_con_list , ignore_index=True)
# Sex_reverse_double_con.to_csv("Sex_reverse_double_con_pol_samp_df.csv")
# 
# Inset here the path to the folder in which the outputs of "sex_reversal_whole_cell_4_connections" exist. 
path_to_files_folder_cells = "Insert path here"
# # whole cell
# Sex_reverse_whole_cell_list = []
# for file in tqdm(range(0,2000)):
#     if path.exists(path_to_files_folder_cells + r"\250422_dimorphic_cells_ID_"+str(file)+".csv"):
#         single_run_df3 = pd.read_csv(path_to_files_folder_cells + r"\250422_dimorphic_cells_ID_"+str(file)+".csv")
#         Sex_reverse_whole_cell_list.append(single_run_df3)    
#     else:
#         print("250422_dimorphic_cells_ID_"+str(file)+".csv"+" was not found")
# 
# Sex_reverse_whole_cell = pd.concat(Sex_reverse_whole_cell_list , ignore_index=True)
# Sex_reverse_whole_cell.to_csv("Sex_reverse_whole_cell_pol_samp_df.csv")
# # =============================================================================
# # ============================================================================                                                                                                                                         

Sex_reverse_single_con = pd.read_csv("Sex_reverse_single_con_pol_samp_df.csv", low_memory=False)
Sex_reverse_double_con = pd.read_csv("Sex_reverse_double_con_pol_samp_df.csv", low_memory=False)
Sex_reverse_whole_cell = pd.read_csv("Sex_reverse_whole_cell_pol_samp_df.csv", low_memory=False)

sensory_voltage_changes = [1.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]

# =============================================================================
# Plots
# =============================================================================
for pol_ind in range(8,16):
    folder = r"saved_graphs\connectivity_perturbations\polarity_index_"+ str(pol_ind)
    dimorphic_connections_head = Sex_reverse_single_con[Sex_reverse_single_con["polarity_index"] == pol_ind]
    double_dimorphic_head = Sex_reverse_double_con[Sex_reverse_double_con["polarity_index"] == pol_ind]
    dimorphic_cells_head = Sex_reverse_whole_cell[Sex_reverse_whole_cell["polarity_index"] == pol_ind]

    # ASH to AVA
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Removing ASH to AVA connectivity \nin hermaphrodites", (0.45, -0.75), dimorphic_connections_head, "no_dimorphic_herm_chem_ASH_to_AVA", "(-) Chemical ASH-AVA", 'cornflowerblue', dimorphic_connections_head, "no_dimorphic_herm_gap_ASH_to_AVA", "(-) Gap ASH-AVA", 'blue', double_dimorphic_head, "no_dimorphic_herm_chem_ASH_AVA_gap_ASH_AVA", "(-) Chemical ASH-AVA, gap ASH-AVA", 'navy')
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Adding ASH to AVA connectivity \nin males",(0.45, -0.75),  dimorphic_connections_head, "fem_male_chem_ASH_to_AVA", "(+) Chemical ASH-AVA", 'salmon', dimorphic_connections_head, "fem_male_gap_ASH_to_AVA", "(+) Gap ASH-AVA", 'red', double_dimorphic_head, "fem_male_chem_ASH_AVA_gap_ASH_AVA", "(+) Chemical ASH-AVA, gap ASH-AVA", 'maroon')

    # dimorphic cells
    cells = ['ASH', 'AVA', 'AVD', 'AVB', 'PVC', 'A', 'B']
    for cell in cells:
        sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\Masculinized and feminized "+ cell, (0.4, -0.75), dimorphic_cells_head, "masc_herm_masculinized_"+cell, "Masculinized "+cell +" (hermaphrodites)", 'blue', dimorphic_cells_head, "fem_male_feminized_"+cell, "Feminized "+cell+" (males)", 'red')

    # dimorphic connectoins
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Hermaphrodites with\na male-specific connection", (0.35, -0.85), dimorphic_connections_head, "masc_herm_chem_A_to_PVC", "(+) Chemical A-PVC", 'cornflowerblue', dimorphic_connections_head, "masc_herm_gap_AVD_to_B", "(+) Gap AVD-B", 'navy', dimorphic_connections_head, "masc_herm_gap_AVB_to_A", "(+) Gap AVB-A", 'blue')
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Males with\na hermaphrodite-specific connection", (0.35, -0.75), dimorphic_connections_head, "fem_male_chem_ASH_to_AVB", "(+) Chemical ASH-AVB", 'maroon', dimorphic_connections_head, "fem_male_gap_AVA_to_AVB", "(+) Gap AVA-AVB", 'salmon')
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Hermaphrodites without\na hermaphrodite-specific connection", (0.35, -0.75), dimorphic_connections_head, "no_dimorphic_herm_chem_ASH_to_AVB",  "(-) Chemical ASH-AVB", 'cornflowerblue', dimorphic_connections_head, "no_dimorphic_herm_gap_AVA_to_AVB", "(-) Gap AVA-AVB", 'navy')
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Males without\na male-specific connection", (0.35, -0.85), dimorphic_connections_head, "no_dimorphic_male_chem_A_to_PVC", "(-) Chemical A-PVC", 'maroon', dimorphic_connections_head, "no_dimorphic_male_gap_AVD_to_B", "(-) Gap AVD-B", 'red', dimorphic_connections_head, "no_dimorphic_male_gap_AVB_to_A", "(-) Gap AVB-A", 'salmon')

    # double dimorphic
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Hermaphrodites with two\nmale-specific connections", (0.4, -0.85), double_dimorphic_head, "masc_herm_chem_A_PVC_gap_AVB_A", "(+) Chemical A-PVC, gap AVB-A", 'navy', 
                                                                                                                                                                                 double_dimorphic_head, "masc_herm_gap_AVD_B_gap_AVB_A", "(+) Gap AVD-B, gap AVB-A", 'blue', 
                                                                                                                                                                                 double_dimorphic_head, "masc_herm_chem_A_PVC_gap_AVD_B", "(+) Chemical A-PVC, gap AVD-B", 'cornflowerblue')
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Males with two\nhermaphrodite-specific connections", (0.5, -1.05), double_dimorphic_head, "fem_male_chem_ASH_AVB_gap_AVA_AVB", "(+) Chemical ASH-AVB, gap AVA-AVB", 'maroon', 
                                                                                                                                                                                 double_dimorphic_head, "fem_male_chem_ASH_AVB_gap_ASH_AVA", "(+) Chemical ASH-AVB, gap ASH-AVA", 'salmon', 
                                                                                                                                                                                 double_dimorphic_head, "fem_male_chem_ASH_AVA_chem_ASH_AVB", "(+) Chemical ASH-AVA, chemical ASH-AVB", 'red', 
                                                                                                                                                                                 double_dimorphic_head, "fem_male_chem_ASH_AVA_gap_AVA_AVB", "(+) Chemical ASH-AVA, gap AVA-AVB", 'darkmagenta',  
                                                                                                                                                                                 double_dimorphic_head, "fem_male_gap_ASH_AVA_gap_AVA_AVB", "(+) Gap ASH-AVA, gap AVA-AVB", 'deeppink')
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Hermaphrodites without two\nhermaphrodite-specific connections", (0.5, -1.05), double_dimorphic_head, "no_dimorphic_herm_chem_ASH_AVB_gap_AVA_AVB", "(-) Chemical ASH-AVB, gap AVA-AVB", 'navy', 
                                                                                                                                                                                             double_dimorphic_head, "no_dimorphic_herm_chem_ASH_AVB_gap_ASH_AVA", "(-) Chemical ASH-AVB, gap ASH-AVA", 'blue', 
                                                                                                                                                                                             double_dimorphic_head, "no_dimorphic_herm_chem_ASH_AVA_chem_ASH_AVB", "(-) Chemical ASH-AVA, chemical ASH-AVB", 'cornflowerblue', 
                                                                                                                                                                                             double_dimorphic_head, "no_dimorphic_herm_chem_ASH_AVA_gap_AVA_AVB", "(-) Chemical ASH-AVA, gap AVA-AVB", 'lightskyblue',  
                                                                                                                                                                                             double_dimorphic_head, "no_dimorphic_herm_gap_ASH_AVA_gap_AVA_AVB", "(-) Gap ASH-AVA, gap AVA-AVB", 'steelblue')
    sf.plot_reverse_probability_per_input(sensory_voltage_changes, dimorphic_connections_head, folder +"\\"+ "Males without two\nmale-specific connections", (0.4, -0.85), double_dimorphic_head, "no_dimorphic_male_chem_A_PVC_gap_AVB_A", "(-) Chemical A-PVC, gap AVB-A", 'maroon', 
                                                                                                                                                                           double_dimorphic_head, "no_dimorphic_male_gap_AVD_B_gap_AVB_A", "(-) Gap AVD-B, gap AVB-A", 'salmon', 
                                                                                                                                                                           double_dimorphic_head, "no_dimorphic_male_chem_A_PVC_gap_AVD_B", "(-) Chemical A-PVC, gap AVD-B", 'red')

title = r"saved_graphs\\overlapping_graphs"
sensory_voltage_changes = [1.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
sf.plot_stacked_reverse_probability(sensory_voltage_changes, [Sex_reverse_whole_cell[Sex_reverse_whole_cell["polarity_index"] == pol_ind] for pol_ind in list(set(Sex_reverse_whole_cell["polarity_index"]))], ["A_vs_B_masc_herm_masculinized_ASH_", "A_vs_B_fem_male_feminized_ASH_"], ["Masculinized ASH (hermaphrodites)", "Feminized ASH (males)"], ["blue", "red"], title + r"\\Masculinized and feminized ASH", (0.45,-0.45))
sf.plot_stacked_reverse_probability(sensory_voltage_changes, [Sex_reverse_whole_cell[Sex_reverse_whole_cell["polarity_index"] == pol_ind] for pol_ind in list(set(Sex_reverse_whole_cell["polarity_index"]))], ["A_vs_B_masc_herm_masculinized_AVA_", "A_vs_B_fem_male_feminized_AVA_"], ["Masculinized AVA (hermaphrodites)", "Feminized AVA (males)"], ["blue", "red"], title + r"\\Masculinized and feminized AVA", (0.45,-0.45))

single_and_double_together = pd.concat([Sex_reverse_single_con, Sex_reverse_double_con], axis = 1)
single_and_double_together = single_and_double_together.loc[:,~single_and_double_together.columns.duplicated()]
sf.plot_stacked_reverse_probability(sensory_voltage_changes, [single_and_double_together[single_and_double_together["polarity_index"] == pol_ind] for pol_ind in list(set(single_and_double_together["polarity_index"]))], ["A_vs_B_no_dimorphic_herm_chem_ASH_to_AVA_", "A_vs_B_no_dimorphic_herm_gap_ASH_to_AVA_", "A_vs_B_no_dimorphic_herm_chem_ASH_AVA_gap_ASH_AVA_"], ["(-) Chemical ASH-AVA", "(-) Gap ASH-AVA", "(-) Chemical ASH-AVA, gap ASH-AVA"], ["cornflowerblue", "cyan", "navy"], title + r"\\Removing ASH to AVA connections in hermaphrodites", (0.45,-0.55))
sf.plot_stacked_reverse_probability(sensory_voltage_changes, [single_and_double_together[single_and_double_together["polarity_index"] == pol_ind] for pol_ind in list(set(single_and_double_together["polarity_index"]))], ["A_vs_B_fem_male_chem_ASH_to_AVA_", "A_vs_B_fem_male_gap_ASH_to_AVA_", "A_vs_B_fem_male_chem_ASH_AVA_gap_ASH_AVA_"], ["(+) Chemical ASH-AVA", "(+) Gap ASH-AVA", "(+) Chemical ASH-AVA, gap ASH-AVA"], ["salmon", "red", "maroon"], title + r"\\Adding ASH to AVA connections in males", (0.45,-0.55))
