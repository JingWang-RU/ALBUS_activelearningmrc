import numpy as np

from .strategy import Strategy


class RandomSampling(Strategy):
	def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
		super(RandomSampling, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)

	def query(self, n, model, tokenizer):
		return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)
