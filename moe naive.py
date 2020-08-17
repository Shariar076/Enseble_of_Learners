import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling,entropy_sampling
from modAL.batch import uncertainty_batch_sampling
from modAL.density import information_density
from sklearn.preprocessing import normalize
import pandas as pd



# train_features, train_labels = make_blobs(n_features=4, n_samples=10000, centers=3, random_state=0, cluster_std=2)
# train_dataset = pd.read_table('./datasets/avila-tr.txt',header=None, sep=",")
train_dataset = pd.read_csv('datasets/Frogs_MFCCs.csv')
train_data = train_dataset.values
train_features = train_data[:, :-4]
train_labels = train_data[:, -1]
#only for manual datasets
lbl_names = np.unique(train_labels)

for i in range(len(lbl_names)):
	train_labels[train_labels==lbl_names[i]] = i
train_labels = train_labels.astype(int)


X_pool = deepcopy(train_features)
y_pool = deepcopy(train_labels)

euc_density = information_density(X_pool, 'euclidean')
train_idx = np.unique(y_pool,return_index=True)[1]

X_train = X_pool[train_idx]
y_train = y_pool[train_idx]

X_pool = np.delete(X_pool, train_idx, axis=0)
y_pool = np.delete(y_pool, train_idx)
euc_density = np.delete(euc_density,train_idx)


learners = []


def density_sampling(classifier, X_pool):
	query_idx = np.argmax(euc_density)
	return query_idx, X_pool[query_idx]

learner = ActiveLearner(
	estimator=RandomForestClassifier(),
	query_strategy=density_sampling,
	X_training=X_train, y_training=y_train
)
learners.append(learner)

knn = KNeighborsClassifier(n_neighbors=3)

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
	query_strategy=entropy_sampling,
	X_training=X_train, y_training=y_train
)
learners.append(learner)



weights = np.ones(shape=len(learners))/len(learners)

n_queries = 4000
loop = 0
strategy_count=np.zeros(len(learners))
x=[]
# while True:
score = []
for learner in learners:
	score.append(learner.score(train_features, train_labels))
unqueried_score = np.min(score)
performance_history = [unqueried_score]
for _ in range(n_queries):
	opinions = []
	learner_id=0
	for learner in learners:
		if learner_id == 1:
			query_idx, query_instance = learner.query(X_pool, n_instances=1)
		else:
			query_idx, query_instance = learner.query(X_pool)  # -> Here

		opinions.append(query_idx)
		learner_id+=1
	opt_idx = np.random.choice(range(len(opinions)),p=weights,size=1,replace=True)[0]
	# print("selected strategy: ", opt_idx)
	strategy_count[opt_idx]+=1
	x.append(opt_idx)
	selected_idx = opinions[opt_idx]
	# print(opinions)
	print("selected Index: ", selected_idx)
	for learner in learners:
		if opt_idx==1:
			learner.teach(
				X=X_pool[selected_idx],
				y=y_pool[selected_idx]
			)
		else:
			learner.teach(
				X=X_pool[selected_idx].reshape(1, -1),
				y=y_pool[selected_idx].reshape(1, )
			)
	X_pool = np.delete(X_pool, selected_idx, axis=0)
	y_pool = np.delete(y_pool, selected_idx)
	euc_density = np.delete(euc_density,selected_idx)

	print("loop: ", _)
	performance_history.append(learners[opt_idx].score(train_features, train_labels))
	# if _> 100:
	del_acc = performance_history[-1]-performance_history[-2]
	weights[opt_idx] += del_acc
	if weights[opt_idx]<0:
		weights[opt_idx] = 0
	weights = normalize([weights], norm='l1')[0]
	# print(weights)
print((strategy_count*100)/n_queries)
print(performance_history)
plt.hist(x)
plt.show()
plt.plot(performance_history)
plt.show()