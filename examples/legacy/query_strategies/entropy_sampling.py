import numpy as np
import torch
from torch.utils.data import Subset

from .strategy import Strategy


class EntropySampling(Strategy):
	def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
		super(EntropySampling, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)

	def query(self, n, model, tokenizer):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(model, tokenizer, Subset(self.dataset_tr, idxs_unlabeled))
		log_probs = torch.log(probs)
		U = (probs * log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]]
