import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from os import path

# Inset here the path to the folder in which the outputs of "interneurons_activations_polarity_8" exist. 
path_to_files_folder = "Insert path here"
# =============================================================================
# Read files from cluster and save to csv
# =============================================================================
# =============================================================================
# interneurons_activations_list = []
# for file in tqdm(range(0,200)):
#     if path.exists(path_to_files_folder + r"\280422_interneurons_activations_ID_"+str(file)+".csv"):
#         single_run_df = pd.read_csv(path_to_files_folder+ r"\280422_interneurons_activations_ID_"+str(file)+".csv")
#         interneurons_activations_list.append(single_run_df)    
#     else:
#         print("280422_interneurons_activations_ID_"+str(file)+".csv"+" was not found")
# 
# interneurons_activations = pd.concat(interneurons_activations_list , ignore_index=True)
# interneurons_activations.to_csv("Interneurons_activations_polarity_8.csv")
# =============================================================================

interneurons_activations = pd.read_csv("Interneurons_activations_polarity_8.csv")

def interneuron_activation_graph(neuron, output_file_name, sensory_voltage_changes, interneurons_activations, lim = False):
    '''This function plots for the neuron of interest its respose the different sensory stimuli across multiple parameter sets. 
    Inputs: 
        1. neuron of interest
        2. name of file (to save the graph)
        3. sensory_voltage_changes - strengths of sensory stimuli
        4. df of interneurons activations (the data to plot)
        5. lim - default = false. whether to make the x and y axes the same. 
    Output: 
        graph (saved and presented)
    '''
    plt.figure(figsize = [6.4, 6.4])
    colors = ["black","dimgrey","darkgrey", "lightgray"]
    legend_elements = [str(int(input_strength * 10**3))+"mV" for input_strength in sensory_voltage_changes]
    neuron_df = interneurons_activations[interneurons_activations.columns[interneurons_activations.columns.str.contains(neuron+"_", case=False)]]
    for sens_input in range(len(sensory_voltage_changes)):
        plt.scatter(interneurons_activations["herm_"+neuron+"_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] * 10 **3, interneurons_activations["male_"+neuron+"_" + str(sensory_voltage_changes[sens_input]* 10**3) +"mV_change"] * 10 **3, color=colors[sens_input], label = legend_elements[sens_input], s=15)
        if lim == True:
            max_change = neuron_df.max().max()
            min_change = neuron_df.min().min()
            margins = 1
            plt.xlim(min_change * 10**3 - margins, max_change * 10**3 + margins)
            plt.ylim(min_change * 10**3 - margins, max_change * 10**3 + margins)
        plt.xlabel("\u0394V " +neuron+ " Hermaphrodite", fontsize = 16)
        plt.ylabel("\u0394V " +neuron+ " Male", fontsize = 16)
        plt.title(neuron +" activations", fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
    plt.legend(loc = 'upper left', fontsize = 14)
    plt.savefig(r"saved_graphs\interneurons\\" + output_file_name + ".pdf")
    plt.show()

sensory_voltage_changes = [75 * 10 **-3, 45 * 10 **-3, 15 * 10 **-3, 1 * 10 **-3]

interneuron_activation_graph("AVA", "AVA activations", sensory_voltage_changes, interneurons_activations, lim = False)
interneuron_activation_graph("AVD", "AVD activations", sensory_voltage_changes, interneurons_activations, lim = False)
interneuron_activation_graph("AVB", "AVB activations", sensory_voltage_changes, interneurons_activations, lim = False)
interneuron_activation_graph("PVC", "PVC activations", sensory_voltage_changes, interneurons_activations, lim = False)
interneuron_activation_graph("AVA", "AVA activations same scale", sensory_voltage_changes, interneurons_activations, lim = True)
interneuron_activation_graph("AVD", "AVD activations same scale", sensory_voltage_changes, interneurons_activations, lim = True)
interneuron_activation_graph("AVB", "AVB activations same scale", sensory_voltage_changes, interneurons_activations, lim = True)
interneuron_activation_graph("PVC", "PVC activations same scale", sensory_voltage_changes, interneurons_activations, lim = True)