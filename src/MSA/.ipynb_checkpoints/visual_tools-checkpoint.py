from datetime import datetime, date
import pandas as pd
import numpy as np


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

def PCA_KMean_