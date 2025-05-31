import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData


def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=2023):

    for res in np.arange(0.2, 2, increment):
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['leiden'].unique()) > n_clusters:
            break
    return res-increment


def leiden(adata, n_clusters, use_rep='emb', key_added='STMCCL', random_seed=2023):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.leiden(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['leiden']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=2023):
    for res in np.arange(0.2, 2, increment):
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['louvain'].unique()) > n_clusters:
            break
    return res-increment


def louvain(adata, n_clusters, use_rep='emb', key_added='STMCCL', random_seed=2023):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.louvain(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['louvain']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata



def mclust_R(adata, n_clusters, use_rep='STMCCL', key_added='STMCCL', random_seed=2024):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import os
    os.environ['R_HOME'] = 'D:/R/R-4.3.3/R-4.3.3'
    os.environ['R_USER'] = 'C:/Users/GQ/.conda/envs/SEDR/Lib/site-packages/rpy2'
    modelNames = 'EEE'
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata
#     import rpy2.robjects as robjects
#     import rpy2.robjects.numpy2ri
#     np.random.seed(random_seed)
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#
#     # Load Mclust
#     robjects.r.library("mclust")
#     rmclust = robjects.r['Mclust']
#
#     # Check and print input data details
#     if use_rep not in adata.obsm:
#         print(f"Error: {use_rep} not found in adata.obsm")
#         return None
#
#     data_to_cluster = adata.obsm[use_rep]
#     print("Input data shape:", data_to_cluster.shape)
#     print("Input data (first few rows):", data_to_cluster[:5, :5])
#
#     # Run Mclust
#     try:
#         res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data_to_cluster), n_clusters, modelNames)
#
#         # Debugging output
#         print("Mclust result type:", type(res))
#         print("Mclust result content:", res)
#
#         if res == robjects.r('NULL'):
#             raise ValueError("Mclust function returned NULL. Check input parameters and data.")
#
#         # Extracting the classification results
#         mclust_res = res.rx2('classification')  # Use rx2 to access R list elements by name
#         print("Mclust classification result type:", type(mclust_res))
#         print("Mclust classification result contents:", mclust_res)
#
#         if mclust_res is None:
#             raise ValueError("Mclust result contains NULL values. Check input parameters and data.")
#
#         # Convert R object to numpy array
#         mclust_res = np.array(mclust_res)
#
#         # Store results in AnnData object
#         adata.obs[key_added] = mclust_res
#         adata.obs[key_added] = adata.obs[key_added].astype('int')
#         adata.obs[key_added] = adata.obs[key_added].astype('category')
#
#         return adata
#     except Exception as e:
#         print("An error occurred while running Mclust:", str(e))
#         return None
# #     try:
# #         print("Starting mclust_R function...")
# #        
# #         print(f"adata shape: {adata.shape}")
# #         print(f"Number of variables (genes): {adata.n_vars}")
# #         print(f"Number of observations (cells): {adata.n_obs}")
# #         print(f"First few var_names: {adata.var_names[:5]}")
# #         print(f"First few obs_names: {adata.obs_names[:5]}")
# #
# #        
# #         
# #         result = [None, np.random.rand(10, 10), np.random.rand(10, 10)]
# #
# #         print("mclust_R function completed successfully.")
# #         return result
# #     except Exception as e:
# #         print("Error in mclust_R:", str(e))
# #         return None
# #
# # def example_function(data):
# #     print("Processing data in example_function...")
# #     print(f"data shape: {data.shape}")
# #    
# #     if isinstance(data, AnnData):
# #         print(f"data X shape: {data.X.shape}")
# #   
# #     return np.random.rand(10, 10)
# #
# # def another_function(data):
# #     print("Processing data in another_function...")
# #     print(f"data shape: {data.shape}")
# #   
# #     return np.random.rand(10, 10)

