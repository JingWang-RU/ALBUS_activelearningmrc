import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pickle
class Strategy:
    def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
        self.dataset_tr = dataset_tr
        self.idxs_lb = idxs_lb
        self.train_fun = train_fun
        self.eval_fun = eval_fun
        self.args = args
        self.n_pool = len(dataset_tr)
        self.T = 1
        self.dup_realindex = []
        self.up_data = []
        self.thresh = args.thresh
        self.num_round = args.num_round
        self.query_learnrate = args.query_learnrate
    def query(self, n, model, tokenizer):
        pass

    def update(self, idxs_lb, dup_realindex=[]):
        self.idxs_lb = idxs_lb
        self.dup_realindex = dup_realindex
        
    def train_norm(self, model, tokenizer, incremental=False, new_pool=None):
        if not incremental:
            idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        else:
            idxs_train = np.arange(self.n_pool)[new_pool]
        return self.train_fun(self.args, Subset(self.dataset_tr, idxs_train), model, tokenizer)

    def train(self, model, tokenizer, incremental=False, new_pool=None):
        if not incremental:
            print("no incremental", flush = True)
            if len(self.dup_realindex) == 0:
                idxs_train = np.arange(self.n_pool)[self.idxs_lb]
                print("org", flush=True)
                
                return self.train_fun(self.args, Subset(self.dataset_tr, idxs_train), model, tokenizer)
            else:
                print("dup", flush=True)
                idxs_train = np.arange(self.n_pool)[self.idxs_lb]
                with open('./scrachidx_dup.pickle','wb') as f:
                    pickle.dump([idxs_train, self.dup_realindex], f)
                realind = np.concatenate([idxs_train, self.dup_realindex])
                num = 0
#                 if len(self.dup_realindex) > 0 :
#                     for ind in self.dup_realindex:
#                         num = num + len(ind)
#                         self.up_data = ConcatDataset([self.up_data, Subset(self.dataset_tr, ind)])
                print("sum of exmaples {} ".format(len(realind)), flush = True)
                return self.train_fun(self.args, Subset(self.dataset_tr, realind), model, tokenizer)
        else:
            print("incremental", flush=True)
            idxs_train = np.arange(self.n_pool)[new_pool]
            if len(self.dup_realindex) == 0:
                print("org", flush=True)
                
                return self.train_fun(self.args, Subset(self.dataset_tr, idxs_train), model, tokenizer)
            else:
                with open('./incincidx_dup.pickle','wb') as f:
                    pickle.dump([idxs_train, self.dup_realindex], f)
                realind = np.concatenate([idxs_train, self.dup_realindex])
            return self.train_fun(self.args, Subset(self.dataset_tr, realind), model, tokenizer)

    def predict_prob(self, model, tokenizer, dataset):
        return self.eval_fun(self.args, model, tokenizer, active=True, eval_dataset=dataset)

    def predict_prob_dropout(self, model, tokenizer, dataset):
        return self.eval_fun(self.args, model, tokenizer, active=True, eval_dataset=dataset,
                             output_type='dropout')

    def predict_prob_dropout_split(self, model, tokenizer, dataset):
        return self.eval_fun(self.args, model, tokenizer, active=True, eval_dataset=dataset,
                             output_type='dropout_split')

    def get_embedding(self, model, tokenizer, dataset):
        return self.eval_fun(self.args, model, tokenizer, active=True, eval_dataset=dataset,
                             output_type='embedding')


