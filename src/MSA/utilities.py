def Return_Matrix_From_Multiprocessing_Output(_path_output):
    """
    Return_Matrix_From_Multiprocessing_Output
    Output: 
    P2_P0	0.2834645669291339
    P3_P0	0.2818791946308725
    ...
    """
    df_tem = pd.read_csv(_path_output, sep='\t', header=None)
    len_df_tem = len(df_tem)
    num_matrix = int(np.sqrt(2*len_df_tem))+1
    print ("N_Pairs: " + str(len_df_tem) + " ~= Matrix Size: " + str(num_matrix))
    
    df_tem_idx = df_tem.loc[:, 0].str.split('_', expand=True)
    matrix_idx = sorted(np.unique(df_tem_idx.values))
    
    df_out_matrix = pd.DataFrame(data = np.diag(np.full(len(matrix_idx), 1.0)),
                                 columns = matrix_idx, index = matrix_idx)
    
    for idx in df_tem_idx.iterrows():
        #print (idx[0], idx[1])
        x = idx[1][0]
        y = idx[1][1]
        score = df_tem.loc[idx[0], 1]
        df_out_matrix.loc[x, y] = score
        df_out_matrix.loc[y, x] = score
        #break
    return df_out_matrix.iloc[0:num_matrix, 0:num_matrix]

import re

def If_contain_letter(index_value):
    return bool(re.search(r'[a-zA-Z]', index_value))

def Negative_Log_Transform_and_Scale(values, scale_min=2, scale_max=20):
    """
    ## edit values scaled between 1 and 10 by rareness weighting
    ## apply a negative log transformation "rarer events = a higher value"
    """
    # Apply the negative log transformation
    transformed_values = -np.log(values)
    # Normalize the transformed values to the range [scale_min, scale_max]
    min_val = np.min(transformed_values)
    max_val = np.max(transformed_values)
    scaled_values = scale_min + (transformed_values - min_val) * (scale_max - scale_min) / (max_val - min_val)
    return -1.0*scaled_values

def Return_Rareness_Weighted_Matrix(_df, w1=-0.5, w2=0.75, w3=0.5):
    #in the manuscript, 𝑤1 = -0.5, 𝑤2 = 0.75, and 𝑤3 = 0.5
    ## Vsub = Vmtc * w1, (w1<=0)
    ## Vins = Vsub * w2 = Vmtc * w1*w2  ([0.5, 1])
    ## Vtns = Vmtc * w3  ([0,1])
    #w1, w2, w3 = -0.5, 0.75, 0.5
    _df.loc[:, "Vmtc"] = Negative_Log_Transform_and_Scale(_df.loc[:, 'count'])
    _df.loc[:, "Vsub"] = w1 * _df.loc[:, "Vmtc"]
    _df.loc[:, "Vins"] = w1*w2 * _df.loc[:, "Vmtc"]
    _df.loc[:, "Vtns"] = w3 * _df.loc[:, "Vmtc"]
    _df.loc["#", :] = 0
    return round(_df,1)#.astype(int)