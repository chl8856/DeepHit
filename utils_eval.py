'''
This provide time-dependent Concordance index and Brier Score:
    - Use weighted_c_index and weighted_brier_score, which are the unbiased estimates.
    
See equations and descriptions eq. (11) and (12) of the following paper:
    - C. Lee, W. R. Zame, A. Alaa, M. van der Schaar, "Temporal Quilting for Survival Analysis", AISTATS 2019
'''

from multiprocessing import Pool, cpu_count
import numpy as np
from lifelines import KaplanMeierFitter


### C(t)-INDEX CALCULATION
def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

### BRIER-SCORE
def brier_score(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)

    return np.mean((Prediction - y_true)**2)

    # result2[k, t] = brier_score_loss(risk[:, k], ((te_time[:,0] <= eval_horizon) * (te_label[:,0] == k+1)).astype(int))


##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G


### C(t)-INDEX CALCULATION: this account for the weighted average for unbaised estimation
# def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
#     '''
#         This is a cause-specific c(t)-index
#         - Prediction      : risk at Time (higher --> more risky)
#         - Time_survival   : survival/censoring time
#         - Death           :
#             > 1: death
#             > 0: censored (including death from other cause)
#         - Time            : time of evaluation (time-horizon when evaluating C-index)
#     '''
#     G = CensoringProb(Y_train, T_train)

#     N = len(Prediction)
#     A = np.zeros((N,N))
#     Q = np.zeros((N,N))
#     N_t = np.zeros((N,N))
#     Num = 0
#     Den = 0
#     for i in range(N):
#         tmp_idx = np.where(G[0,:] >= T_test[i])[0]

#         if len(tmp_idx) == 0:
#             W = (1./G[1, -1])**2
#         else:
#             W = (1./G[1, tmp_idx[0]])**2

#         A[i, np.where(T_test[i] < T_test)] = 1. * W
#         Q[i, np.where(Prediction[i] > Prediction)] = 1. # give weights

#         if (T_test[i]<=Time and Y_test[i]==1):
#             N_t[i,:] = 1.

#     Num  = np.sum(((A)*N_t)*Q)
#     Den  = np.sum((A)*N_t)

#     if Num == 0 and Den == 0:
#         result = -1 # not able to compute c-index!
#     else:
#         result = float(Num/Den)

#     return result

def get_result(n_i_start, n_i_end, N, G, Prediction, T_test, Y_test, Time):
    n_l = n_i_end - n_i_start
    A = np.zeros((n_l,N))
    Q = np.zeros((n_l,N))
    N_t = np.zeros((n_l,N))
    for n_i in range(n_i_start, n_i_end):
        tmp_idx = np.where(G[0,:] >= T_test[n_i])[0]

        if len(tmp_idx) == 0:
            W = (1./G[1, -1])**2
        else:
            W = (1./G[1, tmp_idx[0]])**2

        A[n_i-n_i_start, np.where(T_test[n_i] < T_test)] = 1. * W
        Q[n_i-n_i_start, np.where(Prediction[n_i] > Prediction)] = 1. # give weights

        if (T_test[n_i]<=Time and Y_test[n_i]==1):
            N_t[n_i-n_i_start, :] = 1.

    return (np.sum(((A)*N_t)*Q), np.sum((A)*N_t))


### C(t)-INDEX CALCULATION: this account for the weighted average for unbaised estimation
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    Num_list = []
    Den_list = []

    def update_result_callback(result):
        (num, den) = result
        Num_list.append(num)
        Den_list.append(den)

    def error_handler(e):
        print(e)

    n_processes = cpu_count() - 1
    max_num_per_process = 200
    with Pool(n_processes) as pool:
        results = []
        n_curr = 0
        while n_curr < N:
            for _ in range(n_processes):
                i_start = n_curr
                if n_curr + max_num_per_process >= N:
                    i_end = N
                else:
                    i_end = n_curr + max_num_per_process
                n_curr = i_end
                r = pool.apply_async(get_result, (i_start, i_end, N, G, Prediction, T_test, Y_test, Time), callback=update_result_callback, error_callback=error_handler)
                results.append(r)
            for r in results:
                r.wait()

    Num = np.sum(Num_list)
    Den = np.sum(Den_list)
    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)
    
    return result


# this account for the weighted average for unbaised estimation
def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0,:] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0,:] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
        W[i] = (1. - Y_tilde[i])*float(Y_test[i])/G1 + Y_tilde[i]/G2

    y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W*(Y_tilde - (1.-Prediction))**2)


