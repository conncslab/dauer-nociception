import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def adjust_polarity(Schem, neurons, pol_index, possible_combinations_df):
    '''This function adjusts the polarity of a chemical synapses matrix based on a df with instructions.
       Inputs:
           1. matrix of synaptic strengths
           2. neurons of interest that appear in the matrix
           3. index to use in the polarity combination matrix
           4. polarity combinations matrix (names of columns are the neurons pre and post of each synapse)
       Outputs:
           adjusted matrix of synaptic strengths
    '''
    for synapse in possible_combinations_df.columns:
        Schem[neurons.index(synapse[1]),neurons.index(synapse[0])] = Schem[neurons.index(synapse[1]),neurons.index(synapse[0])] * possible_combinations_df[synapse][pol_index]
    return Schem

def single_flip_change(Schem, neurons, synapse):
    '''This function changes a single connection in chemical synapse to be inhibitory.
       Inputs:
           1. matrix of synaptic strengths
           2. neurons of interest that appear in the matrix
           3. wanted synapse (tuple, name of pre neuron and then post neuron).
       Outputs:
           adjusted matrix of synaptic strengths
    '''
    Schem[neurons.index(synapse[1]),neurons.index(synapse[0])] = Schem[neurons.index(synapse[1]),neurons.index(synapse[0])] * (-1)
    return Schem

def input_current(strength_matrix, t, input_timing, target_neuron, mean_amplitude, sd, *args):
    '''This function creates the input current.
    
       Inputs:
       1. strength_matrix = synaptic strength of chemical synapses or gap junctions, 
          to determine the number of participating neurons.
       2. t = time vector (s).
       3. input_timing = range, the t indexes in which the input will be given.
       4. target_neuron = the index of the neuron that will recieve the input.
       5. mean_amplitude = mean amplitude of the current in Amper.
       6. sd = standard deviation of the current.
       7. optional = for more inputs (different timing or different neurons), repeat 
          sections 3-6.
          
        Output: 
        matrix of input currents, each row is the current of different neurons, each column is a time step.'''
    I = np.zeros((np.size(strength_matrix,0),len(t)))
    I[target_neuron,input_timing] = np.random.normal(mean_amplitude, sd, len(input_timing)) # units: A
    if len(args)>0:
        for i in range(int(len(args)/4)):
            I[args[i*4 + 1],args[i*4]] = np.random.normal(args[i*4 + 2], args[i*4 + 3], len(args[i*4])) 
    return I

def theta(V, Vrest):
    '''This function returns the difference between the inputs if it is positive,
       else it returns zero. The purpose of the function is to make sure a chemical
       synapse would release an output only if the presynaptic neuron experiences 
       depolarization.
       
       Inputs:
       1. np.array of the current voltages
       2. np.array of the resting states (of the same neurons)
       
       Output:
       V - Vrest if V - Vrest > 0
       0 if V - Vrest <= 0'''
    
    theta_res = np.where(V - Vrest > 0, V - Vrest, 0)
    return theta_res

def neurons_elegans(t, Vrest, tm, Rin, gchem, ggap, I, Schem, Sgap, *args):
    '''This function calcultes the change in membrane potential in multilpe neurons,
       following an input current. the neurons can be connected via gap junctions 
       or chemical synapses.
       
       Inputs:
    1. t = time vector (s). the difference between successive numbers must 
       not be larger than tm. 
    2. Vrest = numpy array of the membrane potential (V) at rest of the x examined neurons. 
       should be at size (x,1).
    3. tm = time constant (s) equals Rm * Cm.
    4. Rin = resistance of the neuron (Ω)
    5. gchem = conductance of chemical synapses (S). 
    6. ggap = conductance of gap junctions (S). 
    7. I = numpy array of the input current (A). should be at size (x, t). t- 
       time vector, x- number of neurons.
    8. Schem - matrix of synaptic strength in chemical synapses.
    9. Sgap -  matrix of synaptic strength in gap junctions.
    10.optional - sex. 
    
       Output:
       numpy matrix of the membrane potentials of all neurons along t. each row
       is a single neuron, each column is a time step.'''
    
    V = np.nan * np.empty((np.size(Schem,0),len(t)))
    V[:,0:1] = Vrest
    dec = 0
    for i in range(len(t) - 1):
        if any(V[:,i]> 10 **300) and dec == 0:
           if len(args)<1:
              print("Large number was encountered at t = ",t[i])
           else:
              print("Large number was encountered at t = "+str(t[i])+ " for the "+args[0])
           dec = 1
           V = np.array([[Decimal(x) for x in row] for row in V])
           Sgap = np.array([[Decimal(x) for x in row] for row in Sgap])
           Schem = np.array([[Decimal(x) for x in row] for row in Schem])
           Vrest = np.array([[Decimal(x) for x in row] for row in Vrest])
           I = np.array([[Decimal(x) for x in row] for row in I])
           t = np.array([Decimal(x) for x in t])
           tm = Decimal(tm)
           Rin = Decimal(Rin)
           gchem = Decimal(gchem)
           ggap = Decimal(ggap)
        if i == len(t) - 2:
           V[:,i+1:] = (((((np.dot(Sgap, V[:,i:i+1]) - np.reshape(np.sum(np.multiply(Sgap, np.repeat(V[:,i:i+1],np.size(Sgap,1), axis = 1)),1),(np.size(Sgap,1),1))) * (ggap * Rin) + np.dot(Schem, theta(V[:,i:i+1], Vrest)) * (gchem * Rin) - (V[:,i:i+1] - Vrest) + Rin * I[:,i:i+1])) * (t[i + 1]-t[i])) * 1/tm) + V[:,i:i+1] 
        else:
           V[:,i+1:i+2] = (((((np.dot(Sgap, V[:,i:i+1]) - np.reshape(np.sum(np.multiply(Sgap, np.repeat(V[:,i:i+1],np.size(Sgap,1), axis = 1)),1),(np.size(Sgap,1),1))) * (ggap * Rin) + np.dot(Schem, theta(V[:,i:i+1], Vrest)) * (gchem * Rin) - (V[:,i:i+1] - Vrest) + Rin * I[:,i:i+1])) * (t[i + 1]-t[i])) * 1/tm) + V[:,i:i+1] 
    return V

def basic_conditions_check(V, t, neurons, sensory_start, sensory_end, correction_to_ms):
    '''This function tests the pass\fail status of each set. 
    if the set failed a condition the function returns the failed condition, otherwise it returns True.
    Inputs: 
        1. matrix with the voltage changes in all of the neurons throughout the simulation.
        2. time vector.
        3. participating neurons.
        4. index in t of the start of the sensory stimulation.
        5. index in t of the end of the sensory stimulation.
        6. correction to ms - to find the t that represents the desired ms. 
    Output: 
        list of tuples with the failed conditions\"True" if all of the conditions were met.
    '''
    conditions = []
    if not ((np.mean(V[neurons.index("A"), sensory_start - (1000 * correction_to_ms) : sensory_start])) < (np.mean(V[neurons.index("B"), sensory_start - (1000 * correction_to_ms) : sensory_start]))):
        conditions.append("failed pre forward")
    if not ((np.mean(V[neurons.index("A"), sensory_end - (1000 * correction_to_ms) : sensory_end])) > (np.mean(V[neurons.index("B"), sensory_end - (1000 * correction_to_ms) : sensory_end]))):
        conditions.append("failed post backwards")
    if not (np.max(V)< 100 * 10**-3):
        conditions.append("failed logical range")
    if not (np.all(abs(np.mean(V[:,len(t) - (1000 * correction_to_ms) : len(t)-1], axis = 1) - np.mean(V[:,sensory_start - (1000 * correction_to_ms) : sensory_start - 1], axis = 1)) <= abs(np.mean(V[:,sensory_end - (1000 * correction_to_ms) : sensory_end - 1], axis = 1) - np.mean(V[:,sensory_start - (1000 * correction_to_ms) : sensory_start - 1], axis = 1)))):
        conditions.append("failed return to rest")
    if not conditions:
        conditions = (True)
    else:
        conditions = tuple(conditions)
    return conditions

def cells_activity_conditions_check(V, neurons,sensory_start, sensory_end, correction_to_ms):
    '''This function test the activity of the cells A, B, AVA, and AVB in each set, to see if it fits
    experimental results.
      Inputs: 
        1. matrix with the voltage changes in all of the neurons throughout the simulation.
        2. participating neurons.
        3. index in t of the start of the sensory stimulation.
        4. index in t of the end of the sensory stimulation.
        5. correction to ms - to find the t that represents the desired ms. 
    Output: 
        list of tuples with the failed conditions\"true" if all of the conditions were met.  
    '''
    conditions = []
    if not ((np.mean(V[neurons.index("A"), sensory_start - (1000 * correction_to_ms) : sensory_start])) < (np.mean(V[neurons.index("A"), sensory_end - (1000 * correction_to_ms) : sensory_end]))):
        conditions.append("A not increasing")
    if not ((np.mean(V[neurons.index("AVA"), sensory_start - (1000 * correction_to_ms) : sensory_start])) < (np.mean(V[neurons.index("AVA"), sensory_end - (1000 * correction_to_ms) : sensory_end]))):
        conditions.append("AVA not increasing")
    if not ((np.mean(V[neurons.index("B"), sensory_start - (1000 * correction_to_ms) : sensory_start])) > (np.mean(V[neurons.index("B"), sensory_end - (1000 * correction_to_ms) : sensory_end]))):
        conditions.append("B not decreasing")
    if not ((np.mean(V[neurons.index("AVB"), sensory_start - (1000 * correction_to_ms) : sensory_start])) > (np.mean(V[neurons.index("AVB"), sensory_end - (1000 * correction_to_ms) : sensory_end]))):
        conditions.append("AVB not decreasing")
    if not ((np.mean(V[neurons.index("AVA"), sensory_start - (1000 * correction_to_ms) : sensory_start])) < (np.mean(V[neurons.index("AVB"), sensory_start - (1000 * correction_to_ms) : sensory_start]))):
        conditions.append("AVB not higher forwards")
    if not ((np.mean(V[neurons.index("AVA"), sensory_end - (1000 * correction_to_ms) : sensory_end])) > (np.mean(V[neurons.index("AVB"), sensory_end - (1000 * correction_to_ms) : sensory_end]))):
        conditions.append("AVA not higher backwards")
    if not conditions:
        conditions = (True)
    else:
        conditions = tuple(conditions)
    return conditions

def str_to_nparray(string, wanted_size):
    '''This function returns numpy.arrays that turned into a long string back into numpy.array. 
       Inputs:
           1. the long string.
           2. the wanted size of the numpy array
       Output:
       string as numpy array. 
    '''
    splitted_string = list(string.split(" "))
    filtered_list = list(filter(None, [element.strip("][\n") for element in splitted_string]))
    float_list = [float(i) for i in filtered_list]
    shaped_nparray = np.reshape(float_list, wanted_size)
    return shaped_nparray

def joint_plot_broken_y_axis(t, V, neurons, min_max_up, min_max_down, path_for_graphs, *args):
    '''This function creates a plot of the membrane potentials with broken y axis.
    
       Inputs:
       1. t = time vector (ms).
       2. V = numpy matrix of the membrane potentials of all neurons.
       3. neurons = list of the neurons involved in the simulation.
       4. min_max_up - limits for the upper part of y axis.
       5. min_max_down - limits for the lower part of y axis.
       6. path_for_graphs- path to place to save the data
       7. optional: title. if not inserted a default title will be presented.

       Output:
       plot (saved and presented) 
       
       Taken from:
       https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html
       '''
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.5, 4.5))
    # plot the same data on both axes
#    ax.plot(t * 10 **3, V.T * 10**3)
#    ax2.plot(t * 10 **3, V.T * 10**3)
    ax.plot(t , V.T * 10**3)
    ax2.plot(t , V.T * 10**3)
    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(min_max_up[0], min_max_up[1])  
    ax2.set_ylim(min_max_down[0], min_max_down[1])  

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False, labelsize= 13)  # don't put tick labels at the top
    ax2.tick_params(labelsize= 13)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    f.text(-0.04, 0.5, 'Membrane potential (mV)', va='center', rotation='vertical', fontsize=16)
    plt.xlabel('Time (sec)', fontsize=16)
    ax.legend(neurons, loc='upper right', fontsize = 10.5)

    if len(args)>0:
        title = args[0]
    else:
        title = 'Membrane Potential of Neurons Following an Input Current'
    plt.suptitle(title, fontsize=20) 
    plt.ticklabel_format(useOffset=False)
    plt.xticks(fontsize = 14)
    plt.tight_layout()
    plt.savefig(path_for_graphs +r"\\single set "+title+".pdf", bbox_inches='tight')
    plt.show()

def bar_plot_per_parameter(rlevant_df, Parameters_df, parameters, path_for_graphs):
    '''This function creates for each parameter a histogram of samples values that appered in valid sets.
       Inputs:
       1. rlevant_df - valid sets with a certain polarity
       2. Parameters_df - all possible parameter values
       3. parameters - dictionary, for each parameter specifies the units. 
       4. path_for_graphs - path to place to save the data
       
       Output:
       plot (saved and presented) 
    '''
    plt.figure(figsize=(4, 18))
    mutual_sets = Parameters_df.iloc[rlevant_df["parameters_index"]].copy()
    Parameters_df_full = Parameters_df.copy()
    plt.subplots_adjust(hspace=1.3)
    for n, parameter in enumerate(parameters.keys()):
        ax = plt.subplot(len(parameters),1, n + 1)
        mutual_sets.loc[:,parameter] = mutual_sets.loc[:,parameter].apply(lambda x:x/parameters[parameter][1])
        mutual_sets.loc[:,parameter] = mutual_sets.loc[:,parameter].apply(lambda x:float('%.2g' % x))
        Parameters_df_full.loc[:,parameter] = Parameters_df_full.loc[:,parameter].apply(lambda x:x/parameters[parameter][1])
        Parameters_df_full.loc[:,parameter] = Parameters_df_full.loc[:,parameter].apply(lambda x:float('%.2g' % x))
        counts = mutual_sets[parameter].value_counts()
        for value in set(Parameters_df_full[parameter]):
            if value not in counts.index:
                counts.loc[value] = 0
        counts = counts.sort_index()
        counts.plot(ax=ax, kind = "bar", color = "silver")
        parameter_name = parameter.replace("_", " ")
        ax.set_title(parameter_name, pad = 7, fontsize = 20)
        ax.set_xlabel("Sampled values " + parameters[parameter][0], fontsize = 15)
    plt.savefig(path_for_graphs +r"\\parameters distribution polarity ind " + str(rlevant_df["polarity_index"].iloc[0])+".pdf", bbox_inches='tight')
    plt.show()
    
def plot_reverse_probability_per_input(sensory_voltage_changes, responses_df_male_and_herm, title, anchor, *args):
    '''
    Inputs:
        1. sensory_voltage_changes - list of voltage changes to the sensory neuron that were tested.
        2. responses_df_male_and_herm - df with the WT responses of both sexes
        3. title
        4. anchor - for legend
    args:
    optional. for connectivity perturbations, it is possible to add additional curves with the 
    responses of the masculinized\feminized worms. 
        1. dimorphic_df_change (args[i * 4])
        2. dimorphic_change (args[i * 4 + 1])
        3. label (args[i * 4 + 2])
        4. additional_color (args[i * 4 + 3])
        5. if args > 4, repeat 1-4. 
    
    Output:
       plot (saved and presented) 
    '''
    mean_reversal_herm = []
    mean_reversal_male = []
    for sens_voltage in range(len(sensory_voltage_changes)):
        mean_reversal_herm.append(np.mean(responses_df_male_and_herm["herm_A_vs_B_"+str(sensory_voltage_changes[sens_voltage])+"mV_change"]))
        mean_reversal_male.append(np.mean(responses_df_male_and_herm["male_A_vs_B_"+str(sensory_voltage_changes[sens_voltage])+"mV_change"]))
    plot_margins = 2    
    if len(args)>0:
        dimorphic_df_change = []
        dimorphic_changes = []
        labels = []
        additional_colors =[]
        for i in range(int(len(args)/4)):
            dimorphic_df_change.append(args[i * 4])
            dimorphic_changes.append(args[i * 4 + 1])
            labels.append(args[i * 4 + 2])
            additional_colors.append(args[i * 4 + 3])
            mean_reversal_changed_con = {}
        for change in range(len(dimorphic_changes)):
            mean_reversal_changed_con[labels[change]] = []
            for sens_voltage in range(len(sensory_voltage_changes)):
                mean_reversal_changed_con[labels[change]] += [np.mean(dimorphic_df_change[change]["A_vs_B_"+ dimorphic_changes[change] + "_" +str(sensory_voltage_changes[sens_voltage])+"mV_change"])]
            plt.plot(sensory_voltage_changes,mean_reversal_changed_con[labels[change]], marker = "o", label = labels[change], color = additional_colors[change])
    plt.plot(sensory_voltage_changes,mean_reversal_herm, marker = "o", label = "Hermaphrodites", color = 'sandybrown')
    plt.plot(sensory_voltage_changes,mean_reversal_male, marker = "o", label = "Males", color ='darkturquoise')
    plt.xlim([min(sensory_voltage_changes) - plot_margins, max(sensory_voltage_changes) + plot_margins])# https://seaborn.pydata.org/generated/seaborn.lineplot.html
    plt.ylabel("Predicted Avoidance", fontsize = 16)
    plt.xlabel(r'$\Delta$V in sensory neuron (mV)', fontsize=16)
    if len(args)>0:
        plt.legend(labels + ["Wild-type hermaphrodites","Wild-type males"] , fontsize = 14, loc='lower center', bbox_to_anchor = anchor)       
    else: 
        plt.legend(["Wild-type hermaphrodites","Wild-type males"] , fontsize = 14, loc='lower center', bbox_to_anchor = anchor)               
    if "\\" in title:
       title_on_plot = title.split("\\")[-1]
    else: title_on_plot == title
    plt.title(title_on_plot, fontsize = 18)
    if "\n" in title:
       title = title.replace("\n", "")
    plt.savefig(title +".pdf", bbox_inches='tight')
    plt.show()

def identify_dimorphic(A_herm, A_male, ctype, neurons, sex):
    '''This function finds the dimorphic connections.
        Inputs:
            1. adjacency matrix of hermaphrodites (1 if a connection exits, 0 otherwise)
            2. adjacency matrix of males (1 if a connection exits, 0 otherwise)
            3. connection type (gap or chem)
            4. neurons in the circuit.
            5. sex (are you looking for hermaphrodite specific connections or male's)
        
        Outputs:
            1. list of tuples with the names of the pre and post synaptic cells (in the dimorphic connections)
            2. list of tuples with the indexes of the pre and post synaptic cells (in the dimorphic connections)
        notice, as gap junctions are bi-diractional they appear only once in random order of pre-post.
    '''
    if sex == "herm":
       sex_indices = np.where(A_herm - A_male == 1)
    elif sex == "male":
       sex_indices = np.where(A_herm - A_male == -1)
    else:
        print("please specify one of those sexes: male\herm.")
    sex_cells = []
    sex_ind = []
    for ind1 in range(len(sex_indices[0])):
        sex_ind.append((sex_indices[1][ind1], sex_indices[0][ind1]))
    if ctype == "gap":
       sex_ind = list(set(tuple(sorted(t)) for t in sex_ind))
    for ind2 in range(len(sex_ind)):
        sex_cells.append((neurons[sex_ind[ind2][0]], neurons[sex_ind[ind2][1]]))
    return sex_cells, sex_ind

def initialize_dimorphic_dict(names_for_dict, nrow, ncol):
    '''This function creates an empty dictionary to insert: 
       1. The responses to different stimuli from different sets (if ncol > 1)
       2. the pass\fail status of each set with a change in one dimorphic connection (if ncol == 1).
 
       Inputs:
       1. keys for the dictionary. 
       2. number of expected rows (expected to be the number of tested sets).
       3. number of expected columns - for testing conditions expected to be 1,                 
       otherwise the number of different stimuli. 
         
       Output:
       empty dictionary, the keys are the dimorphic connections (by cells).
     '''
    dimorphic_dict = {}
    for name in names_for_dict:
        if ncol == 1:
            dimorphic_dict[name] = [None] * nrow
        else:
            dimorphic_dict[name] = np.empty((nrow, ncol)) * np.nan
    return dimorphic_dict

def transition_single_con(S_sex1, S_op_sex, ind_pre_copy, ind_post_copy, connection_type):
    '''This function transfers a connection from one sex to the other.
        Inputs:
        1. strength matrix of the original sex.
        2. strength matrix of the opposite sex.
        3. index of the presynaptic cell that will be copied.
        4. index of the post synaptic cell that will be copied.
        5. connection type- chem or gap.
        Output:
        the strength matrix with the trasfered connection.'''
    S_transition = S_sex1.copy()
    S_transition[ind_post_copy, ind_pre_copy] = S_op_sex[ind_post_copy,ind_pre_copy]
    if connection_type == "gap":
       S_transition[ind_pre_copy, ind_post_copy] = S_op_sex[ind_pre_copy, ind_post_copy]
    return S_transition

def single_con_sex_reversal_pol_samp(responses, conditions, sex_ind_type, sex_cells_type, S_sex1, S_sex2, S_other_type, Parameters_df, sensory_start, sensory_end, voltage, sensory_voltage_changes, parameter_ind, con_type, t, Vrest, neurons, I, Dimorphic_connections_Job, row, correction_to_ms):
    '''This function calculates the responses of worms with sex-reversed connction, 
       and checks if the worm still meets all conditions.
       Inputs:
       1. responses- array with the responses of the worm (VA-VB). this array will be editted within the function.
       2. conditions- array with the status if the conditions of the worm (pass\fail). this array will be editted within the function.
       3. sex_ind_type - the indices we want to alter. if it is to remove dimorphic connection the indices need to be of the original sex,
          if it is to add dimorphic connections of the other sex the indices need to be of that other sex.
       4. sex_cells_type - the cells in the dimorphic connection. regarding the sex, same as 3.
       5. S_sex1 - the strength matrix of the original sex, the one we want to sex-reverse. 
       6. S_sex2 - the strength matrix of the opposite sex, from which we want to copy the connections.
       7. S_other_type - the strength matrix of the original sex that is not being altered (if the dimorphic connection
          is a chemical synapse, this will be the gap junctions matrix).
       8. parameters_df - all combinations of biophysical parameters
       9. sensory_start - time of stimulus start
       10.sensory_end - time of stimulus end
       11.voltage - strength of sensory stimulus
       12.sensory_voltage_changes - list of voltage changes of the sensory neuron in response to the stimulus (meaning, different stimuli strengths)
       13.parameter_ind -index of row in parameters_df (from which the parameters are taken).
       14.con_type - gap or chem
       15.t - time steps for simulation
       16.Vrest - resting potential of all cells
       17.neurons - list of neurons in the circuit
       18.I - input current
       19.Dimorphic_connections_Job - from which we take the rest of the parameters.
       20.row - row in Dimorphic_connections_Job.
       21.correction_to_ms - to find the t that represents the desired ms.
       
       The function returns editted responses and conditions'''
    for con in range(len(sex_ind_type)):
        con_name1 = sex_cells_type[con][0] + "_to_" + sex_cells_type[con][1]
        S_sex_reversed = transition_single_con(S_sex1, S_sex2, sex_ind_type[con][0], sex_ind_type[con][1], con_type)
        if con_type == "chem":
            V_sex_reversed = neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, S_sex_reversed, S_other_type)
        elif con_type == "gap":
            V_sex_reversed = neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, S_other_type, S_sex_reversed)
        responses[con_name1][row - Dimorphic_connections_Job.index[0], voltage] = np.mean(V_sex_reversed[neurons.index("A"), sensory_end - (1000 * correction_to_ms):sensory_end]) - np.mean(V_sex_reversed[neurons.index("B"), sensory_end - (1000 * correction_to_ms):sensory_end])
        if sensory_voltage_changes[voltage] == 75 * 10 **-3:
            conditions[con_name1][row - Dimorphic_connections_Job.index[0]] = basic_conditions_check(V_sex_reversed, t, neurons, sensory_start, sensory_end, correction_to_ms)
    return responses, conditions

def double_con_sex_reversal_pol_samp(responses, conditions, cells_double_connections_sex, inds_double_connections_sex, Schem_sex1, Sgap_sex1, Schem_sex2, Sgap_sex2, Dimorphic_double_con_Job, row, t, Vrest, neurons, sensory_voltage_changes, voltage, sensory_start, sensory_end, I, Parameters_df, parameter_ind, correction_to_ms):
    '''This function calculates the responses (VA - VB), and the conditions status (pass\fail) of worms after
       changing two dimorphic connections simultaneously. 
       Inputs:
           1. responses - array with the responses of the worm (VA-VB). this array will be editted within the function.
           2. conditions - array with the status if the conditions of the worm (pass\fail). this array will be editted within the function.
           3. cells_double_connections_sex - the cells in the dimorphic connection. if it is to remove dimorphic connection the connections need to be of the original sex,
              if it is to add dimorphic connections of the other sex the connections need to be of that other sex.
           4. inds_double_connections_sex - the indices we want to alter. regarding the sex, same as 3.
           5. Schem_sex1 - strength matrix of chemical synapses of the original sex (that we want to reverse).
           6. Sgap_sex1 - strength matrix of gap junctions of the original sex (that we want to reverse).
           7. Schem_sex2 - strength matrix of chemical synapses of the opposite sex (from which we want to copy).
           8. Sgap_sex2 - strength matrix of gap junctions of the opposite sex (from which we want to copy).
           9. Dimorphic_double_con_Job - from which we take the rest of the parameters.
           10.row - row in Dimorphic_connections_Job from which the parameters are taken.
           11.t - time steps for simulation
           12.Vrest - resting potential of all cells
           13.neurons - list of neurons in the circuit
           14.sensory_voltage_changes - list of voltage changes of the sensory neuron in response to the stimulus (meaning, different stimuli strengths)
           15.voltage- strength of sensory stimulus
           16.sensory_start - time of stimulus start
           17.sensory_end - time of stimulus end
           18.I - input current
           19.Parameters_df
           20.parameter_ind
           21.correction_to_ms
           
           The function returns editted responses and conditions'''

    for cons1 in range(len(cells_double_connections_sex)):
        cons_name1 = ("_").join(cells_double_connections_sex[cons1][0] + cells_double_connections_sex[cons1][1])
        Schem_reversed = Schem_sex1.copy()
        Sgap_reversed = Sgap_sex1.copy()
        for con in range(len(cells_double_connections_sex[cons1])):
            if inds_double_connections_sex[cons1][con][0] == "chem":
                Schem_reversed = transition_single_con(Schem_reversed, Schem_sex2, inds_double_connections_sex[cons1][con][1], inds_double_connections_sex[cons1][con][2], "chem")
            elif inds_double_connections_sex[cons1][con][0] == "gap":
                Sgap_reversed = transition_single_con(Sgap_reversed, Sgap_sex2, inds_double_connections_sex[cons1][con][1], inds_double_connections_sex[cons1][con][2], "gap")
        V_reversed = neurons_elegans(t, Vrest, Parameters_df.Rm[parameter_ind] * Parameters_df.Capacitance[parameter_ind],  Parameters_df.Rin[parameter_ind], Parameters_df.Chemical_Conductance[parameter_ind], Parameters_df.Gap_Conductance[parameter_ind], I, Schem_reversed, Sgap_reversed)
        responses[cons_name1][row - Dimorphic_double_con_Job.index[0], voltage] = np.mean(V_reversed[neurons.index("A"), sensory_end - (1000 * correction_to_ms) : sensory_end]) - np.mean(V_reversed[neurons.index("B"), sensory_end - (1000 * correction_to_ms) : sensory_end])
        if sensory_voltage_changes[voltage] == 75 * 10 **-3:
            conditions[cons_name1][row - Dimorphic_double_con_Job.index[0]] = basic_conditions_check(V_reversed, t, neurons, sensory_start, sensory_end, correction_to_ms)
    return responses, conditions

def append_to_lists_start(type1, cells_list):
    '''This function appends an element to the begining of all sublists in list.
        Inputs:
        1. string to append
        2. list of tuples
        Output:
        the same list where the first element in every sublist is type1 (the string).'''        
    list1 = []
    for element in cells_list:
        list1.append((type1, element[0], element[1]))
    return list1

def plot_stacked_reverse_probability(sensory_voltage_changes, list_of_responses_df, list_of_responses_coulmns, legend, colors, title, anchor, *args):
    '''
    Inputs:
        1. sensory_voltage_changes - list of voltage changes to the sensory neuron that were tested.
        2. list_of_responses_df - list of dfs with the responses to plot.
        3. list_of_responses_coulmns - list of the columns in the dfs we want to plot.
        4. legend - to describe the columns
        5. colors - same number as in 3,4
        6. title
        7. anchor - for legend
     
    Output:
       plot (saved and presented) 
    '''
    mean_reversals = {df_ind:{group: [] for group in legend} for df_ind in range(len(list_of_responses_df))}
    # For each polarity combination:
    for df_ind in range(len(list_of_responses_df)):
        relevant_df = list_of_responses_df[df_ind]
        # For each group (different sexes\dimorphic changes - columns in df)
        for ind, group in enumerate(legend):
            for sens_voltage in range(len(sensory_voltage_changes)):
                mean_reversals[df_ind][group].append(np.mean(relevant_df[list_of_responses_coulmns[ind]+str(sensory_voltage_changes[sens_voltage])+"mV_change"]))
            plt.plot(sensory_voltage_changes,mean_reversals[df_ind][group], marker = "o", label = group, color = colors[ind], linewidth  = 0.3)
    plot_margins = 2    
    plt.xlim([min(sensory_voltage_changes) - plot_margins, max(sensory_voltage_changes) + plot_margins])# https://seaborn.pydata.org/generated/seaborn.lineplot.html
    plt.ylabel("Predicted Avoidance", fontsize = 16)
    plt.xlabel(r'$\Delta$V in sensory neuron (mV)', fontsize=16)
    plt.legend(legend , fontsize = 14, loc='lower center', bbox_to_anchor = anchor)       
    if "\\" in title:
       title_on_plot = title.split("\\")[-1]
    else: title_on_plot == title
    plt.title(title_on_plot, fontsize = 18)
    if "\n" in title:
       title = title.replace("\n", "")
    plt.savefig(title +".pdf", bbox_inches='tight')
    plt.show()