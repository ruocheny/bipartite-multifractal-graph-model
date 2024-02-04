import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
from networkx import bipartite

import biWMGNumba


def mean_squared_error(A,B):
    return ((np.array(A) - np.array(B))**2).mean()


def KRON(vector,K):
    vec_k = vector
    for k in range(1,K):
        vec_k = np.kron(vec_k,vector)
    return vec_k


def bmg_recon_all(R, l_A, l_B, p, K, ss = 1e-7, EM_iter=100, E_iter=10, M_iter=40, \
                EM_stopping=0.1, M_stopping=0.0001, early_stop_M=True):

    N_A = R.shape[0]
    N_B = R.shape[1]

    M_A = len(l_A)
    M_B = len(l_B)

    # initialization
    l_A_K = KRON(l_A, K)
    l_B_K = KRON(l_B, K)
    tau_A_K = np.repeat([l_A_K], N_A, axis=0) #Use L to initialize tau @ 1st EM loop
    tau_B_K = np.repeat([l_B_K], N_B, axis=0)

    print('initialization: l_A={}, l_B={}, p={}'.format(l_A, l_B, p))


    # start estimation
    llh_all, llh_p_all = [], []
    l_A_all, l_B_all = [], []
    p_all=[]
    tau_A_K_all, tau_B_K_all = [], []

    # EMLOOP
    for EMIdx in range(EM_iter):

        tau_A_K, tau_B_K, l_A, l_B, l_A_K, l_B_K, p_K, p, llh, llh_p \
        = biWMGNumba.EMLOOP(K, E_iter, M_iter, EM_iter, R, l_A, l_B, p, tau_A_K, tau_B_K, 
                            N_A, N_B, M_A, M_B, ss, early_stop_M, M_stopping)
        
        # Append
        llh_all.append(llh)
        llh_p_all.append(llh_p)
        l_A_all.append(l_A)
        l_B_all.append(l_B)
        p_all.append(np.array(p))
        tau_A_K_all.append(tau_A_K)
        tau_B_K_all.append(tau_B_K)
        
        # Print
        print('EM iter={}: l_A={}, l_B={}, p={}, llh={}'.format(EMIdx, l_A, l_B, p, llh))

        if np.isnan(np.array(llh)):
            print('early terminated due to nan')
            return l_A, l_B, p, tau_A_K, tau_B_K, llh_all
        
        # early stopping rule
        if EMIdx >= 20 and abs(llh_all[EMIdx-1]-llh) < EM_stopping:
            break

    return l_A, l_B, p, tau_A_K, tau_B_K, llh_all, l_A_all, l_B_all, p_all, llh_p_all, tau_A_K_all, tau_B_K_all



# BMG: reconstruct graph which keep node prop (with tau)

def graph_recon_bmg(N_A, N_B, node_a, node_b, K, M_A, M_B, p_K, tau_A_K, tau_B_K):

    edgelist_recon = []
    for u in range(N_A):
        for v in range(N_B):
            link_prob = 0
            tmp = np.random.rand()
            for i in range(M_A**K):
                for j in range(M_B**K):
                    link_prob += tau_A_K[u][i]*tau_B_K[v][j]*p_K[i][j]
            if tmp < link_prob:
                edgelist_recon.append((node_a[u],node_b[v]))
                
    B_recon = nx.Graph()
    B_recon.add_nodes_from(node_a, bipartite=0)
    B_recon.add_nodes_from(node_b, bipartite=1)
    B_recon.add_edges_from(edgelist_recon)

    return B_recon



def compute_closeness_centrality_for_one_recon_graph(node_a, node_b, N_A, N_B, B_ori, B_recon, recon_name='Recon'):
    # compute graph metrics: closeness centrality
    close_ori_dict = bipartite.centrality.closeness_centrality(B_ori,node_a,normalized=True)
    close_recon_dict = bipartite.centrality.closeness_centrality(B_recon,node_a,normalized=True)
    
    close_ori_a = [close_ori_dict[n] if n in close_ori_dict else 0 for n in node_a]
    close_recon_a = [close_recon_dict[n] if n in close_recon_dict else 0 for n in node_a]
    
    close_ori_b = [close_ori_dict[n] if n in close_ori_dict else 0 for n in node_b]
    close_recon_b = [close_recon_dict[n] if n in close_recon_dict else 0 for n in node_b]
    
    # ascending sort for error plot
    close_ori_a, idx = np.sort(close_ori_a), np.argsort(close_ori_a)
    close_recon_a = np.array(close_recon_a)[idx]

    close_ori_b, idx = np.sort(close_ori_b), np.argsort(close_ori_b)
    close_recon_b = np.array(close_recon_b)[idx]

    closeness = {'a':{'Ground truth': close_ori_a, recon_name: close_recon_a},
                 'b':{'Ground truth': close_ori_b, recon_name: close_recon_b}}

    return closeness



def compute_degree_for_one_recon_graph(node_a, node_b, N_A, N_B, B_ori, B_recon, recon_name='Recon'):

    deg_ori_dict = bipartite.degrees(B_ori,node_a)
    deg_recon_dict = bipartite.degrees(B_recon,node_a)
    
    deg_ori_a = [deg_ori_dict[1][n] for n in node_a]
    deg_recon_a = [deg_recon_dict[1][n] for n in node_a]
    
    deg_ori_b = [deg_ori_dict[0][n] for n in node_b]
    deg_recon_b = [deg_recon_dict[0][n] for n in node_b]
    
    # ascending sort for error plot
    deg_ori_a, idx = np.sort(deg_ori_a), np.argsort(deg_ori_a)
    deg_recon_a = np.array(deg_recon_a)[idx]

    deg_ori_b, idx = np.sort(deg_ori_b), np.argsort(deg_ori_b)
    deg_recon_b = np.array(deg_recon_b)[idx]

    degree = {'a':{'Ground truth': deg_ori_a, recon_name: deg_recon_a},
              'b':{'Ground truth': deg_ori_b, recon_name: deg_recon_b}}

    return degree


def compute_clustering_coefficient_for_one_recon_graph(node_a, node_b, N_A, N_B, B_ori, B_recon, recon_name='Recon'):
    # compute graph metrics: clustering coefficient
    cc_ori_dict = bipartite.clustering(B_ori)
    cc_recon_dict = bipartite.clustering(B_recon)
    
    cc_ori_a = [cc_ori_dict[n] if n in cc_ori_dict else 0 for n in node_a]
    cc_recon_a = [cc_recon_dict[n] if n in cc_recon_dict else 0 for n in node_a]
    
    cc_ori_b = [cc_ori_dict[n] if n in cc_ori_dict else 0 for n in node_b]
    cc_recon_b = [cc_recon_dict[n] if n in cc_recon_dict else 0 for n in node_b]
    
    # ascending sort for error plot
    cc_ori_a, idx = np.sort(cc_ori_a), np.argsort(cc_ori_a)
    cc_recon_a = np.array(cc_recon_a)[idx]

    cc_ori_b, idx = np.sort(cc_ori_b), np.argsort(cc_ori_b)
    cc_recon_b = np.array(cc_recon_b)[idx]

    cc = {'a':{'Ground truth': cc_ori_a, recon_name: cc_recon_a},
                 'b':{'Ground truth': cc_ori_b, recon_name: cc_recon_b}}

    return cc