import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import Subset

from .strategy import Strategy


class KMeansSampling(Strategy):
	def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
		super(KMeansSampling, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)

	def query(self, n, model, tokenizer):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		embedding = self.get_embedding(model, tokenizer, Subset(self.dataset_tr, idxs_unlabeled))
		embedding = embedding.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(embedding)
		
		cluster_idxs = cluster_learner.predict(embedding)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (embedding - centers) ** 2
		dis = dis.sum(axis=1)
		q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])

		return idxs_unlabeled[q_idxs]
