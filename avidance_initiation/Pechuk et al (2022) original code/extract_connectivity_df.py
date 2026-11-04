import pandas as pd 
import numpy as np
import random 

# =============================================================================
# Description from Cook et al:
# Connectome Adjacency Matrices.
# These sheets give the whole-animal connectomes for the two adult sexes of C. elegans.
# They provide the amount of connectivity between each pair of cells.  
# For chemical connections, rows give the pre-synaptic cell, columns give the 
# post-synaptic cell.  Weights in the body of the matrices are the total number
# of EM serial sections of connectivity, taking into account both the number 
# of synapses and the sizes of synapses.  
# To provide complete coverage of the entire nervous system, the data are 
# assembled from multiple animals and include connections added by 
# extrapolation in gaps where no data were available.  The data in these 
# tables is used in the figures of Cook et al.
# =============================================================================

def check_table(pre_neurons, post_neurons):
    '''This function chechs that the table was read as expected, by checking if the names
       of the neurons appear in the expected column and row. the test is done on ASH (chosen randomly).
       inputs: 
           1. a dataframe column that is expected to contain the pre synaptic neurons (no NA values).
           2. a dataframe row that is expected to contain the post synaptic neurons (no NA values).
       output:
       1 is correct, 0 otherwise.  
    '''
    if sum(pre_neurons.str.contains("ASH", na = False)) and sum(post_neurons.str.contains("ASH", na = False)):
       return 1
    else:
        return 0
    
def organize_connectivity_df(full_connectivity_df, use_last_row = False): 
    '''This function organizes the connectivity matrix such that non existing connection
       will be marked in 0 instead of NA, and that the indices and columns' names will be 
       the pre and postsynaptic neurons (respectively).
       input:
       raw connectivity matrix.
       output:
       organized connectivity matrix. 
    '''
    pre_neurons = full_connectivity_df.iloc[:,2].dropna()
    post_neurons = full_connectivity_df.iloc[2,:].dropna()

    if check_table(pre_neurons, post_neurons):
       print("Good to go")
    else:
       print("Please make sure the dataframe is at the expected format")

    full_connectivity_df = full_connectivity_df.dropna(axis='columns', how = "all")
    if use_last_row == False:
        connectivity_df_by_cells = full_connectivity_df.iloc[3:-1,3:-1]
    else:
        connectivity_df_by_cells = full_connectivity_df.iloc[3:,3:]
    connectivity_df_by_cells = connectivity_df_by_cells.fillna(0)
    connectivity_df_by_cells.index = list(pre_neurons)
    connectivity_df_by_cells.columns = list(post_neurons)
    return connectivity_df_by_cells

def narrow_connectivity_matrix(connectivity_df_by_cells, circuit_neurons):
    '''This function narrows the full connectivity martix to the interest neurons only.
       A connection between two neural classes is the sum of all connections between the classes.
       inputs:
           1. full connectivity matrix.
           2. circuit_neurons - neurons of interest. dictionary where the keys are the names
              of the neurons and the values are the regex expression used to find them in the
              full connectivity matrix. example: {'AVA': 'AVA[RL]'} (appears in right and left form)
       outputs:
           1. small dataframe with the wanted neurons only.
           2. number of neurons found for each key in the dictionay. for example, since AVA appears as 
              both AVAR and AVAL the number will be 2.     
    '''
    narrow_columns_df = pd.DataFrame()
    narrow_df = pd.DataFrame()
    indices = []
    n_neurons_pre = {}
    n_neurons_post = {}
    for k,v in circuit_neurons.items():
        pre_connections = connectivity_df_by_cells.filter(regex=v, axis = 1).agg({'sum'}, axis = 1)
        narrow_columns_df[k] = pre_connections['sum']  
        n_neurons_pre[k] = sum(connectivity_df_by_cells.index.str.count(v))
    for k,v in circuit_neurons.items():
        post_connections = narrow_columns_df.filter(regex=v, axis = 0).agg({'sum'}, axis = 0)
        indices.append(k)
        narrow_df = pd.concat([narrow_df,post_connections])
        n_neurons_post[k] = sum(connectivity_df_by_cells.columns.str.count(v))
    if n_neurons_pre == n_neurons_post:
       n_neurons =  n_neurons_pre
    else:
       print("nunber of pre neurons isn't equal to post, please check")
       n_neurons = n_neurons_pre.update(n_neurons_post)
    narrow_df.index = indices
    return narrow_df, n_neurons

def connectivity_matrix_from_file(herm_chem_full_connectivity, herm_gapj_full_connectivity, male_chem_full_connectivity, male_gapj_full_connectivity, circuit_neurons):
    '''This function organizes the full connectivity dataframe, narrows down the df to the
       neurons of interest only, and makes sure it was done properly (by checking if the number
       of neurons found was the same among males and hermaphrodites for all connection types. for 
       further explanations please look for the functions:
       organize_connectivity_df, narrow_connectivity_matrix. 
    '''
    herm_chem_by_cells = organize_connectivity_df(herm_chem_full_connectivity)
    herm_gapj_by_cells = organize_connectivity_df(herm_gapj_full_connectivity)
    male_chem_by_cells = organize_connectivity_df(male_chem_full_connectivity)
    male_gapj_by_cells = organize_connectivity_df(male_gapj_full_connectivity)

    herm_chem, n_neurons_hc = narrow_connectivity_matrix(herm_chem_by_cells, circuit_neurons)
    herm_gapj, n_neurons_hg = narrow_connectivity_matrix(herm_gapj_by_cells, circuit_neurons)
    male_chem, n_neurons_mc = narrow_connectivity_matrix(male_chem_by_cells, circuit_neurons)
    male_gapj, n_neurons_mg = narrow_connectivity_matrix(male_gapj_by_cells, circuit_neurons)
    
    if n_neurons_hc == n_neurons_hg == n_neurons_mc == n_neurons_mg:
       neurons_numbers = n_neurons_hc
    else:
       print("check number of neurons, unequal")
       neurons_numbers = n_neurons_hc.update(n_neurons_hg, n_neurons_mc, n_neurons_mg)
    return herm_chem, herm_gapj, male_chem, male_gapj, neurons_numbers

def symmetric_Sgap(Sgap):
    '''This function makes the strength matrix of the gap junctions symetric, after
       random changes in the connectivity strength.
       Input:
       Synaptic strength matrix (describing gap junctions).
       output:
       Symmetric synaptic strength matrix (upper triangular is copied to lower triangular).
    ''' 
    Sgap_symmetric = np.triu(Sgap)
    Sgap_symmetric = Sgap_symmetric + Sgap_symmetric.T - np.diag(np.diag(Sgap_symmetric))
    return Sgap_symmetric

def adjusting_strengths(strength_matrix, multiplier_vec, ctype, *args):
    '''This function:
        1. if a threshold was specified (as args) it eliminates every connection which 
           is bellow the threshold. 
        2. adjusts the strength of each connection randomly by multiplying in one of the
           values specified in the multiplier vector. each connection is multiplied by a value
           chosen randomly and independently of the others.
       inputs:
           1. strength_matrix.
           2. multiplier_vec- multiplier vector, with the values that will be used to 
              adjust each connection stregth randomly.
           3. ctype - connections type (gap\chem). for gap connections the returned matrix
              will be symmetric.
           4. optional- threshold.
    '''
    if len(args)>0:
       threshold = args[0]
       strength_matrix = np.where(strength_matrix < threshold, 0, strength_matrix)
    multiplier_matrix = random.choices(multiplier_vec, k = np.size(strength_matrix))
    multiplier_matrix = np.reshape(multiplier_matrix, (np.size(strength_matrix, 0),np.size(strength_matrix,1)))
    if ctype == "gap" or ctype == "Gap" or ctype == "GAP":
        multiplier_matrix = symmetric_Sgap(multiplier_matrix)
    adjusted_matrix = np.multiply(strength_matrix,multiplier_matrix)
    return adjusted_matrix, multiplier_matrix

def normalize_strength_matrix(n_neurons, strength_matrix):
    ''''This function normalizes the strength matrix, so that a connection's strength 
        between two neuronal classes would be the averaged strength of the connections. for example:
        the connection between ASH and AVA is initialy the sum of the following connections: 
        ASHR -> AVAR, ASHL -> AVAR, ASHR -> AVAL, ASHL -> AVAL. we have 4 connection, therefore 
        the strength is divided by 4.
        
        inputs:
            1. number of neurons per class,
            2. strength matrix
        output:
            normalize strength matrix
        ''' 
    n_matrix = np.repeat(n_neurons, len(n_neurons), axis = 0)
    n_transposed = np.transpose(n_matrix)
    normalization_matrix = 1 / (n_matrix * n_transposed)
    normalized_matrix = normalization_matrix * strength_matrix
    return normalized_matrix

def final_connectivity_matrix(herm_chem_full_connectivity, herm_gapj_full_connectivity, male_chem_full_connectivity, male_gapj_full_connectivity, circuit_neurons):
    '''This function:
        1. narrows down the full connectivity matrix to the neurons of interest only.
        2. eliminates chemical connections with strength bellow 10. notice- at this point 
           each connection between 2 neural classes is the sum of all connection between the classes. 
        3. normalizes the connections such that each connection will be the average of all connection 
           between the classes.
        4. eliminates self connections (between a neuron to itself).
        for further explanations please look for the functions:
       connectivity_matrix_from_file, adjusting_strengths, normalize_strength_matrix.  
    '''
    a_herm_chem, a_herm_gapj, a_male_chem, a_male_gapj, neurons_numbers = connectivity_matrix_from_file(herm_chem_full_connectivity, herm_gapj_full_connectivity, male_chem_full_connectivity, male_gapj_full_connectivity, circuit_neurons)
    neurons = list(neurons_numbers.keys())
    n_neurons = np.array([[neurons_numbers[neuron] for neuron in neurons]])

    # threshold for chemical synapses:
    thresh_herm_chem, mult = adjusting_strengths(a_herm_chem,[1], "chem", 10)
    thresh_male_chem, mult = adjusting_strengths(a_male_chem,[1], "chem", 10)

    # normalized strength matrix:
    Sgap_herm_fin = normalize_strength_matrix(n_neurons, a_herm_gapj).T.to_numpy()
    Schem_herm_fin = normalize_strength_matrix(n_neurons, thresh_herm_chem).T
    Sgap_male_fin = normalize_strength_matrix(n_neurons, a_male_gapj).T.to_numpy()
    Schem_male_fin = normalize_strength_matrix(n_neurons, thresh_male_chem).T

    np.fill_diagonal(Sgap_herm_fin, 0)
    np.fill_diagonal(Schem_herm_fin, 0)
    np.fill_diagonal(Sgap_male_fin, 0)
    np.fill_diagonal(Schem_male_fin, 0)
    
    return Sgap_herm_fin, Schem_herm_fin, Sgap_male_fin, Schem_male_fin
