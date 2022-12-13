import numpy as np
from .strategy import Strategy
from torch.utils.data import Subset

import pdb
from functools import reduce
from collections import Counter
from numpy import asarray
from numpy import savetxt
import time
import pickle
class Beyondrc(Strategy):
    def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
        super(Beyondrc, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)

    def query(self, n, model, tokenizer):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        pred_prb = self.predict_prob(model, tokenizer, Subset(self.dataset_tr, idxs_unlabeled))
        print("def query pred_prb ", pred_prb)
        probs_sorted, idxs = pred_prb.sort(descending=True)
        probs = probs_sorted[:, 0] - probs_sorted[:,1]
        max_prob = probs_sorted.max(1)[0].numpy()
        probs = probs.numpy()
        
        query_learnrate = self.query_learnrate
        thresh = self.thresh/(query_learnrate**self.T)
        ind_probs = np.where(probs <= thresh)[0]#0.9
        num_round = self.num_round
        print('T {} orgthre{} thresh {} learnrate {} weak samples {} round {} '.format(self.T, self.thresh, thresh, query_learnrate, len(ind_probs), num_round), flush=True)
        
        if len(ind_probs) >= n:
            probs = probs[ind_probs]
        
        if True:
            num_samples = len(probs)
            b_t = min(probs)
            ind_bt = list(probs).index(b_t)
            mu = num_samples
            T = self.T+1
            regret = np.sqrt(T)
            K = num_samples
            probability = np.zeros((num_samples, ))
            delta = 1e-3
            gamma = np.sqrt(K*T/(regret + np.log(2* np.reciprocal(1.0*delta))))
            
            for i in range(len(probs)):
                if i != ind_bt:
                    p_ta = 1/(mu + gamma*( probs[i]-b_t))
                    probability[i] = (p_ta)
            probability[ind_bt] = 1 - np.sum(probability)
            print("max prob {}".format(probability[ind_bt]), flush=True)
            random_ind = np.random.choice(np.arange(num_samples), n, p=list(probability),replace=False)
            tic = time.time()
            for r in range(num_round):
                random_inds = np.random.choice(np.arange(num_samples), n, p=list(probability),replace=False)
                random_ind = np.concatenate((random_ind, random_inds))
            toc = time.time()
            unique, counts = np.unique(random_ind, return_counts=True)
            
            unique = unique.astype(int)
            prob_counts = counts / sum(counts)
    
            counts_ind = np.argsort(-counts)
            cindex = unique[counts_ind][:n]
            new_counts = counts[counts_ind][:n]
            for i in range(len(cindex)):
                print("gap {} prob_0 {} ind {} count {} ".format( probs[i], max_prob[i], cindex[i], new_counts[i]), flush=True)
            acc_sel = 1
            if ind_bt in cindex:
                print('till now {} / {} best been selected '.format(acc_sel, self.T + 1), flush=True)
            
        if len(ind_probs) >= n:
            real_index = ind_probs[cindex]
        else:
            real_index = cindex
        realkey = [idxs_unlabeled[ind] for ind in real_index]
        
        dup_realindex = dict(zip(realkey, np.subtract(new_counts, 1)))
        result_ind = []
        if sum(dup_realindex.values()) > 0:
            result_ind = np.concatenate( [np.repeat(key, dup_realindex[key]) for key in dup_realindex.keys()  if dup_realindex[key]>0] )

        return idxs_unlabeled[real_index], result_ind
