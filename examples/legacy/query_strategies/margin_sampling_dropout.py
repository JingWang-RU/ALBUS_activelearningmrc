import numpy as np
import torch
from torch.utils.data import Subset

from .strategy import Strategy


class MarginSamplingDropout(Strategy):
	def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
		super(MarginSamplingDropout, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)

	def query(self, n, model, tokenizer):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout(model, tokenizer, Subset(self.dataset_tr, idxs_unlabeled))
		probs_sorted, idxs = probs.sort(descending=True)
		U = probs_sorted[:, 0] - probs_sorted[:, 1]
		return idxs_unlabeled[U.sort()[1][:n]]
