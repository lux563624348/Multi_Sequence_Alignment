## components for Clustering Algorithm
from src.MSA.__init__ import *
from src.MSA.levenshtein import *
from src.MSA.utilities import *

def Initate_Dict_For_Cluster(_centroid_ids):
    """
    For given centrod ids, create a dict as {#_cluster: ID}, 
    {0: ['P40'], 1: ['P72'], 2: ['P48'], 3: ['P50'], 4: ['P67']}
    """
    centroid_ids = _centroid_ids
    dict_cluster_meta = dict()
    for i in range(len(centroid_ids)):
        dict_cluster_meta[i]=[]
        dict_cluster_meta[i].append(centroid_ids[i])
        #dict_cluster_meta[i].remove('P50')
    return dict_cluster_meta

def Return_Max_Key(data):
    """
    Finds the key associated with the maximum value in the given dictionary.
    Parameters:
    data (dict): A dictionary where keys are tuples and values are numeric.
    Returns:
    tuple: The key associated with the maximum value.
    float: The maximum value.
    """
    if not data:
        raise ValueError("The dictionary is empty")

    # Initialize variables
    max_value = float('-inf')
    max_key = None

    # Iterate through each key-value pair in the dictionary
    for key, value in data.items():
        if value > max_value:
            max_value = value
            max_key = key

    return max_key, max_value

def Find_Key_For_Value(data, target_value):
    """
    Finds the key associated with the specified value in the dictionary.

    Parameters:
    data (dict): A dictionary where keys are integers and values are lists.
    target_value (str): The value to search for in the lists within the dictionary.

    Returns:
    int: The key associated with the target value.
    None: If the target value is not found.
    """
    if not data:
        raise ValueError("The dictionary is empty")

    # Iterate through each key-value pair in the dictionary
    for key, value_list in data.items():
        if target_value in value_list:
            return key

    return None  # Return None if the target_value is not found

def Find_Max_From_Dict(_dict):
    """# Find the key with the maximum value"""
    max_key = max(_dict, key=_dict.get)
    max_value = _dict[max_key]
    return max_key, max_value

def Assign_Cluster_On_Max_Similarity(_filename, _dict_cluster_meta, IDs, centroid_ids, df_groups, dict_rareness_matrix):
    """
    Assign_Cluster_On_Max_Similarity, only use dict, no pandas
    Input: 
        IDs to be assigned
        centroid_ids
        df_groups
    """

    for new_id in IDs:
        if (new_id in centroid_ids):
            print (new_id, " exist, continue")
        else:
            dict_similarity = dict()
            df_new = df_groups.get_group(new_id).loc[:, [col_seq, col_time]].drop_duplicates()
            for cid in centroid_ids:
                name_pair = str(new_id)+'_'+str(cid)
                df_centroid = df_groups.get_group(cid).loc[:, [col_seq, col_time]].drop_duplicates()
                similarity = Main_Compute_Similarity_For_Pair_and_Save(_filename, name_pair, 
                                    df_new, df_centroid, dict_rareness_matrix)
                
                dict_similarity[(new_id, cid)] = similarity  ## first new_id, then centrod id 
            #break
        max_key, max_value = Find_Max_From_Dict(dict_similarity)  ## max_key: first new_id, then centrod id
        highest_cluster = Find_Key_For_Value(_dict_cluster_meta, max_key[1])
        _dict_cluster_meta[highest_cluster].append(max_key[0])
    return _dict_cluster_meta


def Calculate_Pairwise_Similarity_For_A_Cluster(_filename, _ids, _df_groups, _dict_rareness_matrix):
    """
    ## for a given list of ids and df_groups, calculate pairwise similarity
    """
    dict_rareness_weighted = _dict_rareness_matrix
    list_similarity = []

    for i in range(len(_ids)):
        if (100*i/len(_ids) % 10 == 0):
            print ("Process Patient: ", 100*i/len(_ids), '%')
        p1 = _ids[i]
        for j in range(i):
            if (i!=j):
                p2 = _ids[j]
                name_pair = str(p1)+'_'+str(p2)
                df_p1 = _df_groups.get_group(p1).loc[:, [col_seq, col_time]].drop_duplicates()
                df_p2 = _df_groups.get_group(p2).loc[:, [col_seq, col_time]].drop_duplicates()
                similarity = Main_Compute_Similarity_For_Pair_and_Save(_filename, name_pair, 
                                df_p1, df_p2, dict_rareness_weighted)
                list_similarity.append([name_pair, similarity])
    return list_similarity

def Return_Centroid_Patient_For_Kmean_Cluster(_cluster, _df_out):
    return _df_out[_df_out.k_means_cluster == _cluster].patient_id_C.values[0]

def Return_Patients_For_Kmean_Cluster(_cluster, _df_out_meta):
    return _df_out_meta[_df_out_meta.k_means_cluster == _cluster].patient_id.values
    
def Return_Substr_Match(_substr, _df):
    # Define the substring to search for
    search_substring = str(_substr)
    mask = _df.apply(lambda x: x.astype(str).str.contains(search_substring))
    rows_with_substring = _df[mask.any(axis=1)]
    return rows_with_substring


""" This is conflict with  Calculate_Pairwise_Similarity_For_A_Cluster()
import pandas as pd
def Return_Pair_Similarity_Within_A_Cluster(_cluster, _df_out, _df_out_meta, _path_all_pair):
    df_out = _df_out
    pID_C = Return_Centroid_Patient_For_Kmean_Cluster(_cluster, df_out)
    df_all_pair_simiarity = pd.read_csv(_path_all_pair, sep='\t', header=None)
    df_substr_similarity = Return_Substr_Match(pID_C, df_all_pair_simiarity)
    patients_from_cluster = Return_Patients_For_Kmean_Cluster(_cluster, _df_out_meta)
    patients_from_cluster_No_Centroid = [x for x in patients_from_cluster if x != pID_C]
    df_centroid_similarity = pd.DataFrame()
    for pID in patients_from_cluster_No_Centroid:
    #    print (pID, Return_Substr_Match(pID, df_substr_similarity).shape)
        df_centroid_similarity = pd.concat([df_centroid_similarity, Return_Substr_Match(pID, df_substr_similarity)], ignore_index = True)
        #break
    return df_centroid_similarity
"""

