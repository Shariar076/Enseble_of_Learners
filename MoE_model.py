import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling,entropy_sampling,margin_sampling
from modAL.batch import uncertainty_batch_sampling
from modAL.density import information_density
from sklearn.preprocessing import normalize,StandardScaler
import timeit
import pandas as pd
from sklearn.datasets import load_breast_cancer


# train_features, train_labels = make_blobs(n_features=4, n_samples=600, centers=3, random_state=0, cluster_std=2)
# train_dataset = pd.read_table('./datasets/avila-tr.txt',header=None, sep=",")
# train_dataset = pd.read_csv('datasets/Frogs_MFCCs.csv')
# train_dataset = pd.read_csv('datasets/creditcard.csv')
train_data = np.load('./datasets/checkerboard4x4_train.npz')
# train_data = load_breast_cancer()
# train_data = load_wine()
# print(len(train_data['data']))
# cl1_data = train_dataset.loc[train_dataset.iloc[:, -1] == 0]
# cl2_data = train_dataset.loc[train_dataset.iloc[:, -1] == 1]
# train_dataset = pd.concat([cl1_data.iloc[:800], cl2_data])
# train_data = train_dataset.values
# train_features = train_data[:, :-1]
# train_labels = train_data[:, -1]
# train_features = train_data['data']
# train_labels = train_data['target']

train_features = train_data['x']
train_features = StandardScaler().fit_transform(train_features)
train_labels = train_data['y']
train_labels = np.ravel(train_labels)
# only for manual datasets
# lbl_names = np.unique(train_labels)
#
# for i in range(len(lbl_names)):
# 	train_labels[train_labels==lbl_names[i]] = i
# train_labels = train_labels.astype(int)
#
nExp=1
perf_history_all=[]
x = []
for exp in range(nExp):
	start0 = timeit.default_timer()
	X_pool = deepcopy(train_features)
	y_pool = deepcopy(train_labels)

	euc_density = information_density(X_pool, 'euclidean')
	print("Exp no", exp)
	# train_idx = np.unique(y_pool,return_index=True)[1]
	# print("trsin idx: ",train_idx)
	# train_idx = np.append(train_idx , [0])
	train_idx = np.random.choice(range(X_pool.shape[0]), size=300, replace=False)
	X_train = X_pool[train_idx]
	y_train = y_pool[train_idx]

	X_pool = np.delete(X_pool, train_idx, axis=0)
	y_pool = np.delete(y_pool, train_idx)
	euc_density = np.delete(euc_density,train_idx)

	learners = []


	def density_sampling(classifier, pool):
		# euc_density = information_density(pool, 'euclidean')
		query_idx = np.argmax(euc_density)
		return query_idx, pool[query_idx]


	knn = KNeighborsClassifier(n_neighbors=3)

	learner = ActiveLearner(
		estimator=knn,
		query_strategy=density_sampling,
		X_training=X_train, y_training=y_train
	)
	learners.append(learner)

	learner = ActiveLearner(
		estimator=RandomForestClassifier(),
		query_strategy=density_sampling,
		X_training=X_train, y_training=y_train
	)
	learners.append(learner)

	learner = ActiveLearner(
		estimator=knn,
		query_strategy=uncertainty_batch_sampling,
		X_training=X_train, y_training=y_train
	)
	learners.append(learner)

	learner = ActiveLearner(
		estimator=RandomForestClassifier(),
		query_strategy=uncertainty_sampling,
		X_training=X_train, y_training=y_train
	)
	learners.append(learner)

	learner = ActiveLearner(
		estimator=RandomForestClassifier(),
		query_strategy=margin_sampling,
		X_training=X_train, y_training=y_train
	)
	learners.append(learner)

	learner = ActiveLearner(
		estimator=RandomForestClassifier(),
		query_strategy=entropy_sampling,
		X_training=X_train, y_training=y_train
	)
	learners.append(learner)
	start = timeit.default_timer()
	# print("time: ", start - start0)

	model = RandomForestClassifier()
	model.fit(X_train, y_train)

	weights = np.ones(shape=len(learners))/len(learners)

	n_queries =200
	loop = 0
	strategy_count=np.zeros(len(learners))

	# while True:

	print(np.shape(y_pool))
	unqueried_score = model.score(train_features,train_labels)
	print("initial score: ",unqueried_score)
	performance_history = [unqueried_score]
	for _ in range(n_queries):

		learner_id = np.random.choice(range(len(learners)),p=weights,size=1,replace=True)[0]
		# print("selected strategy: ", learner_id)
		strategy_count[learner_id]+=1
		x.append(learner_id)

		if learner_id == 2:
			query_idx, query_instance = learners[learner_id].query(X_pool, n_instances=3)

				# raise ValueError
			# print(query_idx)
		else:
			query_idx, query_instance = learners[learner_id].query(X_pool)  # -> Here

		train_idx= np.append(train_idx, query_idx)

		model = model.fit(train_features[train_idx],train_labels[train_idx])
		# selected_idx = opinions[opt_idx]
		# print(opinions)
		# print("selected Index: ", query_idx)
		for learner in learners:
			if learner_id==2:
				learner.teach(
					X=X_pool[query_idx],
					y=y_pool[query_idx]
				)
			else:
				learner.teach(
					X=X_pool[query_idx].reshape(1, -1),
					y=y_pool[query_idx].reshape(1, )
				)
		X_pool = np.delete(X_pool, query_idx, axis=0)
		y_pool = np.delete(y_pool, query_idx)
		euc_density = np.delete(euc_density,query_idx)

		# print("loop: ", _)
		performance_history.append(model.score(train_features, train_labels))
		# if _> 100:
		del_acc = performance_history[-1]-performance_history[-2]
		weights[learner_id] += del_acc*1.0
		if weights[learner_id]<0:
			weights[learner_id] = 0
		weights = normalize([weights], norm='l1')[0]
	# print(weights)
	perf_history_all.append(performance_history)
	end = timeit.default_timer()
	print("time: ", end - start)
# print((strategy_count*100)/n_queries)
perf_history_all = np.array(perf_history_all).reshape(nExp, n_queries+1)
np.savetxt('perf_history_all.txt',perf_history_all)
np.savetxt('x.txt',x)

#This part in local machine
performance_history = np.loadtxt('perf_history_all.txt')
performance_history = np.mean(performance_history,axis=0)
# x = np.loadtxt('x.txt')
# plt.hist(x)
# plt.show()
plt.plot(performance_history)
plt.show()