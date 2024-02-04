import numpy as np
import random
import math
import numba as nb
from numba import jit,njit
from numba.typed import List
import numpy as np

#basic func √
@njit
def KRON(vector,K):
    vec_k = vector
    for k in range(1,K):
        vec_k = np.kron(vec_k,vector)
    return vec_k

@njit
def DecConver(Mo,Ko):
    #qo_list = [] # 0->K-1 K-bits
    qo_list = List()
    for qo in range(0,pow(Mo,Ko)):
        #qo_list_item = []
        qo_list_item = List()
        for ko in range(0,Ko):
            qo_list_item.append(qo % Mo)
            qo = int(qo / Mo)
        qo_list.append(qo_list_item)
    return qo_list,len(qo_list)


# E step
@njit
def E_STEP(K, tau_A_k, tau_B_k, l_A_k, l_B_k, p_k, R, N_A, N_B, M_A, M_B):
    
    tau_A_0 = np.zeros((N_A, pow(M_A,K)))
    tau_B_0 = np.zeros((N_B, pow(M_B,K)))

    for u in range(N_A):
        for q in range(pow(M_A,K)):
            for v in range(N_B):
                for h in range(pow(M_B,K)):
                    tmp = R[u][v] * np.log(p_k[q][h]) + (1 - R[u][v]) * np.log(1 - p_k[q][h])
                    # import pdb; pdb.set_trace()
                    tau_A_0[u][q] += tau_B_k[v][h] * tmp
                    tau_B_0[v][h] += tau_A_k[u][q] * tmp
    for u in range(N_A):
        tau_A_0_max = tau_A_0[u,:].max()
        tau_A_0[u,:] -= np.ones((pow(M_A,K))) * tau_A_0_max
    for v in range(N_B):
        tau_B_0_max = tau_B_0[v,:].max()
        tau_B_0[v,:] -= np.ones((pow(M_B,K))) * tau_B_0_max
        
    #tau_A_0 = np.repeat([l_A_k], N_A, axis=0) * np.exp(tau_A_0)
    #tau_B_0 = np.repeat([l_B_k], N_B, axis=0) * np.exp(tau_B_0)
    tau_A_0 = l_A_k.repeat(N_A).reshape(N_A,-1) * np.exp(tau_A_0)
    tau_B_0 = l_B_k.repeat(N_B).reshape(N_B,-1) * np.exp(tau_B_0)

    # Normalization
    #tau_A_0 /= np.tile(np.sum(tau_A_0, axis=1).reshape(N_A, 1), (1, pow(M_A,K)))
    #tau_B_0 /= np.tile(np.sum(tau_B_0, axis=1).reshape(N_B, 1), (1, pow(M_B,K)))
    for u in range(0,N_A):
        tau_A_0[u] /= tau_A_0[u].sum()
    for v in range(0, N_B):
        tau_B_0[v] /= tau_B_0[v].sum()

    return tau_A_0, tau_B_0 # N_A * Pow(M_A,K)


# M step√
@njit
def M_STEP(K,Q_list,H_list, p, tau_A_k, tau_B_k, R, N_A, N_B, M_A, M_B, ss):

    p_k = KRON(p,K)
    
    #likelihood for p ######
    # second term in llh (L_THETA)
    llh_p = 0 
    for u in range(N_A):
        for v in range(N_B):
            for q in range(M_A ** K):
                for h in range(M_B ** K):
                        llh_p += tau_A_k[u][q] * tau_B_k[v][h] \
                         * (R[u][v] * np.log(p_k[q][h]) + (1 - R[u][v]) * np.log(1 - p_k[q][h]))
    
    #gradient of p
    for i in range(M_A):
        for j in range(M_B):
            gradient = 0 #REAL Gradient
            for q in range(pow(M_A,K)):
                for h in range(pow(M_B,K)):
                    # term1
                    term1 = 0
                    for u in range(N_A):
                        for v in range(N_B):
                            term1 += tau_A_k[u][q] * tau_B_k[v][h] \
                        * (R[u][v]/p_k[q][h] - (1 - R[u][v]) / (1 - p_k[q][h]))
                            
                    # term2--> can be accerated here
                    term2 = p_k[q][h]/p[i][j]
                    indi = 0
                    for k in range(K):
                        if Q_list[q][k] == i and H_list[h][k] == j:
                            indi += 1
                    term2 = term2 * indi
                    # Gradient
                    gradient += term1 * term2
            #Update p[i][j] as O/P
            p[i][j] += ss * gradient

    return p, llh_p

# calculate object function√
@njit
def L_THETA(K, llh_p, l_A_k, l_B_k, p, tau_A_k, tau_B_k, R, N_A, N_B, M_A, M_B):
    #l_A_k = KRON(l_A,K) 
    #l_B_k = KRON(l_B,K)
    #llh_l_A = np.sum(tau_A_k * np.repeat([np.log(l_A_k)], N_A, axis=0), axis=(0,1))
    #llh_l_B = np.sum(tau_B_k * np.repeat([np.log(l_B_k)], N_B, axis=0), axis=(0,1))
    x_A = np.log(l_A_k)
    xx_A = tau_A_k * x_A.repeat(N_A).reshape(N_A,-1)
    llh_l_A = xx_A.sum(axis=0).sum(axis=0)
    x_B = np.log(l_B_k)
    xx_B = tau_B_k * x_B.repeat(N_B).reshape(N_B,-1)
    llh_l_B = xx_B.sum(axis=0).sum(axis=0)
    
    #llh_tau_A = np.sum(tau_A_k * np.log(tau_A_k), axis=(0,1))
    #llh_tau_B = np.sum(tau_B_k * np.log(tau_B_k), axis=(0,1))
    #y_A = tau_A_k * np.log(tau_A_k)
    y_A = np.log(tau_A_k**tau_A_k)
    llh_tau_A = y_A.sum(axis=0).sum(axis=0)
    y_B = np.log(tau_B_k**tau_B_k)
    llh_tau_B = y_B.sum(axis=0).sum(axis=0)
    
    l_theta = llh_l_A + llh_l_B + llh_p - llh_tau_A - llh_tau_B
    
    #import pdb; pdb.set_trace()
    
    return l_theta


# EM algorithm√
# @njit
def EMLOOP(K, E_iter, M_iter, EM_iter, R, l_A, l_B, p, tau_A_k, tau_B_k, N_A, N_B, M_A, M_B, ss, early_stop, M_stopping):
    #import pdb; pdb.set_trace()
    l_A_k = KRON(np.array(l_A),K)
    l_B_k = KRON(np.array(l_B),K)
    p_k = KRON(p,K)

    for _ in range(E_iter):    
        tau_A_k, tau_B_k = E_STEP(K, tau_A_k, tau_B_k, l_A_k, l_B_k, p_k, R, N_A, N_B, M_A, M_B)
    
    # import pdb; pdb.set_trace()
    
    # M step L #K
    Q_list,len_Q = DecConver(M_A,K) #q_0,q_1,q_2
    H_list,len_H = DecConver(M_B,K)
    l_A = [1/(N_A*K)] * M_A #create list, length = M_A
    l_B = [1/(N_B*K)] * M_B
    #
    for i in range(M_A):
        term3 = 0
        for u in range(N_A):
            for q in range(pow(M_A,K)):
                indi_a = 0
                for k in range(K):
                    if Q_list[q][k] == i:
                        indi_a += 1
                term3 += tau_A_k[u][q] * indi_a
        l_A[i] *= term3
    #
    for j in range(M_B):
        term4 = 0
        for v in range(N_B):
            for h in range(pow(M_B,K)):
                indi_b = 0
                for k in range(K):
                    if H_list[h][k] == j:
                        indi_b += 1
                term4 += tau_B_k[v][h] * indi_b
        l_B[j] *= term4           
    # 
    #llh_p_all = []
    llh_p_all = List()

    # M step loop => Count P
    for mIdx in range(M_iter):
        p, llh_p = M_STEP(K,Q_list,H_list, p, tau_A_k, tau_B_k, R, N_A, N_B, M_A, M_B, ss) #update p
        llh_p_all.append(llh_p)
        # print(llh_p)

        # early stopping rule
        if early_stop:
            # if mIdx >= 10 and abs(llh_p_all[-1]-llh_p_all[-2])/abs(llh_p_all[-1]) < M_stopping:
            if mIdx >= 10 and abs(llh_p_all[-1]-llh_p_all[-2]) < M_stopping:
                break
                
    llh = L_THETA(K, llh_p, l_A_k, l_B_k, p, tau_A_k, tau_B_k, R, N_A, N_B, M_A, M_B)
    
    return tau_A_k, tau_B_k, l_A, l_B, l_A_k, l_B_k, p_k, p, llh, llh_p_all
