import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling,entropy_sampling,margin_sampling
from modAL.batch import uncertainty_batch_sampling
from modAL.density import information_density
import pandas as pd
import timeit
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# train_features, train_labels = make_blobs(n_features=4, n_samples=600, centers=3, random_state=0, cluster_std=2)
train_dataset = pd.read_table('./datasets/avila-tr.txt',header=None, sep=",")
# train_dataset = pd.read_csv('datasets/Frogs_MFCCs.csv')
# train_dataset = pd.read_csv('datasets/creditcard.csv')
# print(train_dataset.shape)
# train_data = np.load('./datasets/checkerboard4x4_train.npz')
# train_data = np.load('./datasets/gaussian_clouds.npz')
# train_data = load_breast_cancer()
# train_data = load_wine()
# print(len(train_data['data'][0]))
# cl1_data = train_dataset.loc[train_dataset.iloc[:, -1] == 0]
# cl2_data = train_dataset.loc[train_dataset.iloc[:, -1] == 1]
# train_dataset = pd.concat([cl1_data.iloc[:800], cl2_data])
train_data = train_dataset.values
train_features = train_data[:, :-1]
train_labels = train_data[:, -1]
# train_features = train_data['data']
# train_labels = train_data['target']

# train_features = train_data['x']
# train_features = StandardScaler().fit_transform(train_features)
# train_labels = train_data['y']
# train_labels = np.ravel(train_labels)
# only for manual datasets
lbl_names = np.unique(train_labels)

for i in range(len(lbl_names)):
	train_labels[train_labels==lbl_names[i]] = i
train_labels = train_labels.astype(int)

strategy =2

perf_history_all = []
nExp=1

print("Strategy: ", strategy)
for exp in range(nExp):
	X_pool = deepcopy(train_features)
	y_pool = deepcopy(train_labels)

	euc_density = information_density(X_pool, 'euclidean')
	print("Exp: ", exp)
	# train_idx = np.unique(y_pool,return_index=True)[1]
	# print("train idx",train_idx)
	# train_idx = np.append(train_idx , [0])

	train_idx = np.random.choice(range(X_pool.shape[0]), size=300, replace=False)

	X_train = X_pool[train_idx]
	y_train = y_pool[train_idx]

	X_pool = np.delete(X_pool, train_idx, axis=0)
	y_pool = np.delete(y_pool, train_idx)
	euc_density = np.delete(euc_density,train_idx)


	def density_sampling(classifier, X):
		# euc_density = information_density(X_pool, 'euclidean')
		query_idx = np.argmax(euc_density)
		return query_idx, X[query_idx]


	start = timeit.default_timer()
	knn = KNeighborsClassifier(n_neighbors=3)
	if strategy==1:
		learner = ActiveLearner(
			estimator=knn,
			query_strategy=density_sampling,
			X_training=X_train, y_training=y_train
		)
	if strategy==2:
		learner = ActiveLearner(
			estimator=RandomForestClassifier(),
			query_strategy=density_sampling,
			X_training=X_train, y_training=y_train
		)
	if strategy==3:
		learner = ActiveLearner(
			estimator=knn,
			query_strategy=uncertainty_batch_sampling,
			X_training=X_train, y_training=y_train
		)
	if strategy==4:
		learner = ActiveLearner(
			estimator=RandomForestClassifier(),
			query_strategy=uncertainty_sampling,
			X_training=X_train, y_training=y_train
		)
	if strategy==5:
		learner = ActiveLearner(
			estimator=RandomForestClassifier(),
			query_strategy=margin_sampling,
			X_training=X_train, y_training=y_train
		)
	if strategy==6:
		learner = ActiveLearner(
			estimator=RandomForestClassifier(),
			query_strategy=entropy_sampling,
			X_training=X_train, y_training=y_train
		)

	# model = RandomForestClassifier()
	# model.fit(X_train, y_train)

	unqueried_score = learner.score(train_features, train_labels)
	# print("initial score: ", unqueried_score)
	performance_history = [unqueried_score]
	n_queries = 200

	for _ in range(n_queries):

		if strategy==3:
			query_idx, query_instance = learner.query(X_pool,n_instances= 3)  # -> Here
			learner.teach(
				X=X_pool[query_idx],
				y=y_pool[query_idx]
			)
		else:
			query_idx, query_instance = learner.query(X_pool)
			learner.teach(
				X=X_pool[query_idx].reshape(1,-1),
				y=y_pool[query_idx].reshape(1, )
			)

		# train_idx= np.append(train_idx, query_idx)
		# learner.fit(train_features[train_idx],train_labels[train_idx])
		# print("loop: ", _)

		performance_history.append(learner.score(train_features, train_labels))
		X_pool = np.delete(X_pool, query_idx, axis=0)
		y_pool = np.delete(y_pool, query_idx)
		euc_density = np.delete(euc_density, query_idx)


	perf_history_all.append(performance_history)
	end = timeit.default_timer()
	print("time: ", end - start)
# print((strategy_count*100)/n_queries)
perf_history_all = np.array(perf_history_all).reshape(nExp, n_queries + 1)
np.savetxt('perf_history_all'+str(strategy)+'.txt', perf_history_all)
print("strategy: ", strategy)

# This part in local machine
performance_history = np.loadtxt('perf_history_all'+str(strategy)+'.txt')
performance_history = np.mean(performance_history, axis=0)
#
plt.plot(performance_history)
plt.show()