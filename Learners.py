import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling,entropy_sampling
from modAL.batch import uncertainty_batch_sampling
from modAL.density import information_density
from sklearn.preprocessing import StandardScaler

# train_features, train_labels = make_blobs(n_features=4, n_samples=1000, centers=4, random_state=0, cluster_std=1.5)
# train_dataset = pd.read_table('./datasets/avila-tr.txt',header=None, sep=",")
# train_dataset = pd.read_csv('datasets/Frogs_MFCCs.csv')
# train_dataset = pd.read_csv('datasets/creditcard.csv')
# cl1_data = train_dataset.loc[train_dataset.iloc[:, -1] == 0]
# cl2_data = train_dataset.loc[train_dataset.iloc[:, -1] == 1]
#     # print len(cl1_data)
#     # print len(cl2_data)
# train_dataset = pd.concat([cl1_data.iloc[:20000], cl2_data])
# train_data = train_dataset.values
train_data = np.load('./datasets/checkerboard2x2_train.npz')
# train_features = train_data[:, :-1]
# train_labels = train_data[:, -1]
train_features = train_data['x']
train_features = StandardScaler().fit_transform(train_features)
train_labels = train_data['y']
train_labels = np.ravel(train_labels)
# print(np.shape(train_data))
# print(np.shape(train_labels))
# #only for manual datasets
# lbl_names = np.unique(train_labels)
#
# for i in range(len(lbl_names)):
# 	train_labels[train_labels==lbl_names[i]] = i
# train_labels = train_labels.astype(int)


X_pool = deepcopy(train_features)
y_pool = deepcopy(train_labels)

# euc_density = information_density(X_pool, 'euclidean')
print("density done")
train_idx = np.unique(y_pool,return_index=True)[1]

X_train = X_pool[train_idx]
y_train = y_pool[train_idx]

X_pool = np.delete(X_pool, train_idx, axis=0)
y_pool = np.delete(y_pool, train_idx)
# euc_density = np.delete(euc_density,train_idx)


learners = []


def density_sampling(classifier, X):
	euc_density = information_density(X_pool, 'euclidean')
	query_idx = np.argmax(euc_density)
	return query_idx, X[query_idx]

knn = KNeighborsClassifier(n_neighbors=2)
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


# knn = KNeighborsClassifier(n_neighbors=3)

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

ind =0

total_perf=[]


for learner in learners:
	X=deepcopy(X_pool)
	y=deepcopy(y_pool)
	unqueried_score = learner.score(train_features, train_labels)
	# unqueried_score = score(learner, train_features, train_labels)
	performance_history = [unqueried_score]
	n_queries = 200
	loop = 0
	# while True:
	for _ in range(n_queries):
		if ind == 2:
			query_idx, query_instance = learner.query(X, n_instances=1)
		# print(query_idx)
		else:
			query_idx, query_instance = learner.query(X)  # -> Here

		if ind==2:
			learner.teach(
				X=X[query_idx],
				y=y[query_idx]
			)
		else:
			learner.teach(
				X=X[query_idx].reshape(1, -1),
				y=y[query_idx].reshape(1, )
			)

		print("loop: ", ind)
		performance_history.append(learner.score(train_features, train_labels))
		# performance_history.append(score(learner(train_features, train_labels))
		# remove queried instance from pool
		X = np.delete(X, query_idx, axis=0)
		y = np.delete(y, query_idx)
		# euc_density = np.delete(euc_density, query_idx)
		# if performance_history[-1]>0.95:
		# 	break
		# loop+=1

	total_perf.append(performance_history)
	# plt.subplot(2, 2, ind+1)
	# plt.plot(performance_history)
	ind+=1

np.savetxt('total_perf.txt',total_perf)

#this part in local machine
total_perf = np.loadtxt('total_perf.txt')
plt.figure(figsize=(2 * 10, 10))
for i in range(len(total_perf)):
	plt.subplot(3, 3, i+1)
	plt.plot(total_perf[i])
plt.savefig('curve_of_learner.png')
