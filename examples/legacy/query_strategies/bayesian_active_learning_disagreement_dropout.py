import numpy as np
import torch
from torch.utils.data import Subset

from .strategy import Strategy


class BALDDropout(Strategy):
	def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
		super(BALDDropout, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)

	def query(self, n, model, tokenizer):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout_split(model, tokenizer, Subset(self.dataset_tr, idxs_unlabeled))
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1
		return idxs_unlabeled[U.sort()[1][:n]]
