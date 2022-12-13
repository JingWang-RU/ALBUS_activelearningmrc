import numpy as np
import torch
from torch.utils.data import Subset

from .strategy import Strategy


class LeastConfidenceDropout(Strategy):
	def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
		super(LeastConfidenceDropout, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)
		
	def query(self, n, model, tokenizer):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout(model, tokenizer, Subset(self.dataset_tr, idxs_unlabeled))
		U = probs.max(1)[0]
		return idxs_unlabeled[U.sort()[1][:n]]
