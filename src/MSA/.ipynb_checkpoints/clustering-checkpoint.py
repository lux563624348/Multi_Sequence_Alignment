## components for Clustering Algorithm
## require pandas
import pandas as pd
from src.MSA.levenshtein import *
from src.MSA.utilities import *

def Find_Max_From_Dict(_dict):
    # Find the key with the maximum value
    max_key = max(_dict, key=_dict.get)
    max_value = _dict[max_key]
    return max_key, max_value

def Get_Sequence_Dates_From_pID(_pid, df_groups, _len = -1):
    col_code = 'event'
    col_date = 'date'
    date_format = '%Y-%m-%d'
    df_seq1 = df_groups.get_group(_pid).loc[:, [col_code, col_date]].sort_values(col_date).drop_duplicates()
    if (_len != -1):
        df_seq1 = df_seq1.iloc[0:_len, :]
    seq1 = [x for x in df_seq1.loc[:, col_code]]
    date1 = [x for x in df_seq1.loc[:, col_date]]
    return df_seq1, seq1, date1

def Assign_Cluster_For_Max_Similarity(_pID, _df_groups, _df_out, _dict_score, _max_transposition_date, _filename):
    """
    For given patient ID, assign to a cluster basing on max similarity
    """
    df_seq1, seq1, date1 = Get_Sequence_Dates_From_pID(_pID, _df_groups)
    col_code = "dx"
    col_date = "service_dt"
    similarity_dict = dict()
    for pIDc in _df_out.patient_id_C:
        df_seq_c, seq_c, date_c = Get_Sequence_Dates_From_pID(pIDc, df_groups)
        similarity = Main_Compute_Similarity_For_Given_Two_Patients(df_seq_c, df_seq1, 
                                                        col_code, col_date, _dict_score, _max_transposition_date)
        similarity_dict[(pIDc, _pID)] = similarity
        pair_name = str(pIDc)+"_"+str(_pID)
        with open(_filename, 'a') as f:
            f.write(pair_name +'\t'+ str(round(similarity, 3))+'\n')
            time.sleep(0.001)
        #break
    max_key, max_value = Find_Max_From_Dict(similarity_dict)
    pID_max_cluster = _df_out[_df_out.patient_id_C == max_key[0]].k_means_cluster.values[0]
    return pID_max_cluster

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

def Calculate_Pairwise_Similarity_For_A_Cluster(_filename, _ids, _df_groups, _dict_rareness_matrix, _max_transposition_date):
    ## for a given list of ids and df_groups, calculate pairwise similarity
    dict_rareness_weighted = _dict_rareness_matrix
    list_similarity = []
    col_time = 'date'
    col_seq = 'event'
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
                                df_p1, df_p2, col_seq, col_time, dict_rareness_weighted, _max_transposition_date)
                list_similarity.append([name_pair, similarity])
    return list_similarity