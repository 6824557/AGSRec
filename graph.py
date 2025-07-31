import torch.nn
import copy
import random
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse.linalg import eigsh
from multiprocessing import Process, Queue
def get_graph_matrix(file,flag,buildgraphweight):
    if flag==1:
        graph_matrix = np.loadtxt(file+'graph_matrixi'+str(buildgraphweight)+'.csv', delimiter=',')
    else:
        graph_matrix = np.loadtxt(file + 'graph_matrix2.csv', delimiter=',')
    return graph_matrix


def get_graph_nodes_id2idx(file,flag,buildgraphweight):
    node_id2idx = {}
    if flag == 1:
        with open(file+"graph_node_id2idxi"+str(buildgraphweight)+".txt", 'r') as f:
            for line in f:
                node_id, node_idx = line.split(',')
                node_id2idx[int(node_idx)] = int(node_id)
    else:
        with open(file+"graph_node_id2idx2.txt", 'r') as f:
            for line in f:
                node_id, node_idx = line.split(',')
                node_id2idx[int(node_idx)] = int(node_id)
    return node_id2idx
def load_graph_node_features(path, feature1='check_count', feature2='poi_cat_id', feature3='latitude', feature4='longitude'):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()
    return X
def get_graph_node_features(path,flag,buildgraphweight):
    if flag==1:
        path = path+"graph_nodesi"+str(buildgraphweight)+".csv"
    else:
        path = path+"graph_nodes2.csv"
    graph_node_features = load_graph_node_features(
        path,
        "check_count",
        "poi_cat_id",
        "latitude",
        "longitude")
    return graph_node_features

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')