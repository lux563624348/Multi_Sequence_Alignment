from datetime import datetime, date
import pandas as pd
import numpy as np

import simple_icd_10_cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance
## plot
import matplotlib.pyplot as plt
import seaborn as sns
import umap 
import umap.plot

def Return_Des_For_Code(_seq):
    df_icd10 = pd.read_csv("")
    des = []
    for dx in _seq:
        des.append(df_icd10.loc[dx, "Description"])
    return '; '.join(des)

def PCA_for_DF(_df_tem):
    """ PCA for dataframe"""
    df_tem = _df_tem
    return None

def PCA_Kmean_Plot(num_cluster = 3):
    return None

def Umap_as_Pdf(_path, _df_matrix):
    plt.savefig(_path, format = 'pdf')
    return None


def FindCD10Code(_dx_cd):
    ## ICD10,   https://en.wikipedia.org/wiki/ICD-10
    ###  International Statistical Classification of Diseases and Related Health Problems (ICD), 
    ### a medical classification list by the World Health Organization (WHO)

    # https://pypi.org/project/icd10-cm/
    if cm.is_valid_item(_dx_cd):
        description_code = cm.get_description(_dx_cd)
   # if icd10.exists(_dx_cd):
        #description_code = icd10.find(_dx_cd).description  
    else: 
        description_code = "NA"
    return description_code

def Return_Des_For_DX(_seq):
    #df_icd10 = pd.read_csv("./data/ICD-10_Table.tsv", sep='\t', index_col = "Code")
    descriptions = []
    for dx in _seq:
        descriptions.append(FindCD10Code(dx))
    return '; '.join(descriptions)

def PCA_for_DF(df_tem, num_pca = 3):
    """PCA for dataframe"""
    df_2 = df_tem
    #### Remove unicode of list in python 
    df2_gene_id = df_2.index #[x.encode('ascii', 'ignore') for x in df_2.index.values]
    
    #### In here we set targets as the name of columns, which means our purpose is to compare \
    #### the relationship between different columns.
    targets=df_2.columns #[x.encode('ascii', 'ignore') for x in df_2.columns]
    df_2_T = df_2.transpose()
    # Separating out the features
    x = df_2_T.loc[ :, df2_gene_id ].values
    # Separating out the target
    y = df_2_T.loc[targets,:].values
    # Standardizing the featuresbio
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=num_pca)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = [f"PC{i+1}" for i in range(num_pca)] )
    ###########reindex df with df2, just using following command.
    principalDf.index = df_2_T.index
    principalDf=principalDf.sort_index()
    targets =  principalDf.index #[x.encode('ascii', 'ignore') for x in principalDf.index]
    print ("PCA Number of Components:" + str(pca.n_components_)  )
    print ("PCA Explained Variance Ratio: " + str(pca.explained_variance_ratio_) )  
    return principalDf, pca.explained_variance_ratio_

def generate_colormap(N):
    # Generate a colormap with N distinct colors
    colormap = plt.get_cmap('gist_ncar', N+2)  # You can choose different colormaps
    colors = colormap(np.linspace(0, 1, N+2))
    return colors

def PCA_KMean_Plot(df_matrix, num_clsuter=3, title='T60_20_patients'):
    """PCA Kmean and return df patient & cluster"""
    Parameters_Detail = title
    kmeans = KMeans(n_clusters=num_clsuter)
    kmeans.fit(df_matrix)
    
    principalDf, pca_ratio = PCA_for_DF(df_matrix)
    principalDf = principalDf.loc[df_matrix.index.astype(str), :]
    ### PLOT
    fig = plt.figure(figsize = (6,5))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1_'+Parameters_Detail, fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('Exp Var Ratio \n PC1 = ' + str(round(pca_ratio[0],3))+
                 '  &  PC2 = '+str(round(pca_ratio[1],3)) +
                 '  &  PC3 = '+str(round(pca_ratio[2],3)), fontsize = 20)
    ax.scatter(principalDf.PC1, principalDf.PC2, c=kmeans.labels_)
    ax.grid()
    fig.savefig('PCA_'+Parameters_Detail+"_"+str(num_clsuter)+"_center_Kmean"+'.png')
    ax.grid(False)
    #sns.heatmap(df_matrix.apply(lambda x: round(x, 1)), annot=True, ax = ax)
    #ax.set_title(Parameters_Detail)
    #fig = ax.get_figure()
    #fig.savefig('Heatmap_'+Parameters_Detail+"_"+str(num_clsuter)+'.png')
    df_kmeans = pd.DataFrame(data = {'pID': df_matrix.index.astype(str), 'Cluster': kmeans.labels_})
    return df_kmeans

def Output_Umap_As_PDF(_path_out, _df_matrix, _meta_cluster):
    embedding = umap.UMAP(n_components=2, random_state=0)
    scaler = StandardScaler()
    df_matrix = _df_matrix
    clusters = _meta_cluster
    unique_clusters = np.unique(clusters)
    
    matrix_scaled = scaler.fit_transform(df_matrix.values)
    matrix_umap = embedding.fit_transform(matrix_scaled)
    # Create a scatter plot with custom figure size
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    # Scatter plot with legend
    #color_map
    colos = generate_colormap(len(unique_clusters))
    for i in range(len(unique_clusters)):
        cluster_indices = np.where(clusters == unique_clusters[i])
        plt.scatter(matrix_umap[cluster_indices, 0], matrix_umap[cluster_indices, 1],
                    label=f'{unique_clusters[i]}', s=10, alpha=0.7, c = colos[i])
    # Add legend
    plt.legend(title='Clusters', bbox_to_anchor=(1.0, 1.0), loc='lower right', ncol=round(len(unique_clusters))/3)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.savefig(_path_out.replace(".txt", '_UMAP.pdf'), format='pdf')
    return None

def PCA_KMean_Plot_New(_path_out, _principalDf, _pca_ratio, kmeans, num_clsuter=3, title='T10_20_patients'):
    """PCA Kmean and return df patient & cluster"""
    Parameters_Detail = title
    principalDf, pca_ratio =  _principalDf, _pca_ratio #PCA_for_DF(df_matrix)
    principalDf = principalDf.loc[df_matrix.index.astype(str), :]
    ### PLOT
    fig = plt.figure(figsize = (6,5))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1_'+Parameters_Detail, fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('Exp Var Ratio \n PC1 = ' + str(round(pca_ratio[0],3))+
                 '  &  PC2 = '+str(round(pca_ratio[1],3)) +
                 '  &  PC3 = '+str(round(pca_ratio[2],3)), fontsize = 20)
    ax.scatter(principalDf.PC1, principalDf.PC2, c=kmeans.labels_)
    ax.grid()
    fig.savefig(_path_out.replace(".txt", '_PCA_'+Parameters_Detail+"_"+str(num_clsuter)+"_center_PCA.pdf"), format='pdf')
    ax.grid(False)
    return None

## Patient history cluster Table
def Generate_Sum_Patients_Cluster(df_matrix_meta, df_groups): 
    sum_p_history = []
    col_cluster = "k_means_cluster"
    num_patients = len(df_matrix_meta)
    df_patient_groups = df_matrix_meta.groupby(col_cluster)
    df_groups = df_groups
    for order, n_Cluster in df_patient_groups:
        percent = np.round(100* (len(n_Cluster)/ num_patients), 1)
        list_sequence = []
        for pID in n_Cluster.patient_id:
            df_seq, seq, date = Get_Sequence_Dates_From_pID(pID, df_groups)
            list_sequence.append(seq)
        sum_p_history.append([order, percent, list_sequence, 'Des'])
    
    df_sum_p_history = pd.DataFrame(data = sum_p_history, columns = [col_cluster, "%", "Sequences", "Description"])
    df_sum_p_history.sort_values("%", ascending = False).to_csv("Sum_Patient_Cluster.tsv", sep='\t', index=None)
    return None

def Kmean_Output_Centroid(_path_matrix, _n_cluster, _df_groups):
    """This is a combination of PCA_KMean_Plot_New & Generate_Sum_Patients_Cluster"""
    global df_matrix
    df_matrix = pd.read_csv(_path_matrix, sep='\t', index_col='Unnamed: 0').T
    df_matrix.T.to_csv(_path_matrix.replace(".txt", '_Declassified.tsv'), sep='\t', index=None, header=False)
    df_matrix_PCA_meta = pd.DataFrame(data = df_matrix.index.astype(str), columns = ["patient_id"])

    
    df_groups = _df_groups
    n_cluster = _n_cluster
    kmeans = KMeans(n_clusters=n_cluster)

    ## use 10 PCAs for KMeans 
    df_matrix_PCA, pca_ratio = PCA_for_DF(df_matrix, 10)
    kmeans.fit(df_matrix_PCA)
    
    df_matrix_PCA_meta = pd.DataFrame(data = {'patient_id': df_matrix_PCA.index.astype(str), 'k_means_cluster': kmeans.labels_})
    df_matrix_PCA_meta.to_csv(_path_matrix.replace(".txt", '_meta.tsv'), sep='\t', index=None)
    
    #PCA_KMean_Plot_New(_path_matrix, df_matrix_PCA, pca_ratio,  kmeans, n_cluster, "T_10")
    ## also save UMAP
    Output_Umap_As_PDF(_path_matrix, df_matrix_PCA, df_matrix_PCA_meta.k_means_cluster)
    

    cluster_centers = kmeans.cluster_centers_
    
    # Calculate the distances from each point to each cluster center
    distances = distance.cdist(df_matrix_PCA, cluster_centers, 'euclidean')
    closest_indices = np.argmin(distances, axis=0)
    closest_points = df_matrix_PCA.iloc[closest_indices, :]
    #print("Cluster Centers:\n", cluster_centers)
    #print("Closest Data Points:\n", closest_points.index)
    
    cluster_pindex = closest_points.index.astype(int)
    cluster_label = kmeans.predict(df_matrix_PCA.iloc[closest_indices, :])
    
    list_sequence = []
    sum_p_history = []
    for pID, label in zip(cluster_pindex, cluster_label):
        df_seq, seq, date = Get_Sequence_Dates_From_pID(pID, df_groups)
        list_sequence.append(seq)
        des = Return_Des_For_DX(seq)
        sum_p_history.append([label, pID, seq, des])
    df_sum_p_history = pd.DataFrame(data = sum_p_history, columns = ['k_means_cluster','patient_id_C', 'dx_C', "description_C"])
    #df_sum_p_history.to_csv("Sum_" + str(n_cluster) + "_Cluster_Centroid.tsv", sep='\t', index=None)

    ## prepare seqs & % of each cluster
    col_cluster = "k_means_cluster"
    sum_patients_history = []
    df_patient_groups = df_matrix_PCA_meta.groupby(col_cluster)
    for order, n_Cluster in df_patient_groups:
        percent = np.round(100* (len(n_Cluster)/ len(df_matrix_PCA_meta)), 1)
        list_sequence = []
        for pID in n_Cluster.patient_id:
            df_seq, seq, date = Get_Sequence_Dates_From_pID(pID, df_groups)
            list_sequence.append(seq)
        sum_patients_history.append([order, percent, list_sequence, 'Des'])
    
    df_sum_patients_history = pd.DataFrame(data = sum_patients_history, columns = [col_cluster, "%", "Sequences", "Description"])
    
    df_out = df_sum_p_history.merge(df_sum_patients_history, on = col_cluster, how = 'outer')
    ### output format for final _path_matrix.replace(".txt", '_Centroid_Sum.tsv')
    df_out.to_csv(_path_matrix.replace(".txt", '_Sum_Kmean_Cluster_with_Centroid.tsv'), sep='\t', index=None)
    return df_out

def Display_pIDs(IDs, df_groups):
    for p1ID in IDs:
        df_seq1, seq1, date1 = Get_Sequence_Dates_From_pID(p1ID, df_groups)
        p1 = Person(p1ID)
        p1.set_story(df_seq1.drop_duplicates(), 'dx', 'service_dt')
        p1.visualize_history()
    return None
    
def Get_Sequence_Dates_From_pID(_pid, df_groups_patients):
    pID = int(_pid)
    col_code = "dx"
    col_date = "service_dt"
    date_format = '%Y-%m-%d'
    df_groups = df_groups_patients 
    df_seq1 = df_groups.get_group(pID).loc[:, [col_code, col_date]].sort_values(col_date).drop_duplicates()
    seq1 =  [x for x in df_seq1.loc[:, col_code].values]
    date1 = [datetime.strptime(dat_str, date_format).date() for dat_str in df_seq1.loc[:, col_date].values]
    return df_seq1, seq1, date1