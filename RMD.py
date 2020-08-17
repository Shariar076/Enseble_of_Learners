import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.density import information_density
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

# train_dataset = pd.read_table('./datasets/avila-tr.txt',header=None, sep=",")
X, y = make_blobs(n_features=2, n_samples=1000, centers=3, random_state=0, cluster_std=0.7)
train_data = load_iris()#train_dataset.values
train_features = X#train_data['data']#train_data[:, :-1]
train_labels = y#train_data['target']#train_data[:, -1]
#only for manual datasets
# lbl_names = np.unique(train_labels)
# for i in range(len(lbl_names)):
#     train_labels[train_labels==lbl_names[i]] = i
# train_labels = train_labels.astype(int)

X_pool = deepcopy(train_features)
y_pool = deepcopy(train_labels)
# cosine_density = information_density(X_pool, 'euclidean')

train_idx = np.unique(y_pool,return_index=True)[1] #instead of takin g randomly take one sample of each class IMP!!!

X_train = X_pool[train_idx]
y_train = y_pool[train_idx]
# print(y_train,train_idx)

# creating a reduced copy of the data with the known instances removed
X_pool = np.delete(X_pool, train_idx, axis=0)
y_pool = np.delete(y_pool, train_idx)
# cosine_density = np.delete(cosine_density, train_idx)


def density_sampling(classifier, X_pool):
	cosine_density = information_density(X_pool, 'euclidean')
	query_idx = np.argmax(cosine_density)
	return query_idx, X_pool[query_idx]


learner = ActiveLearner(
	estimator=RandomForestClassifier(),
	query_strategy=density_sampling,
	X_training=X_train, y_training=y_train
)

unqueried_score = learner.score(train_features, train_labels)
performance_history = [unqueried_score]
n_queries = 500
loop = 0
for _ in range(n_queries):

	query_idx, query_instance = learner.query(X_pool)  # -> Here
	# print(query_instance, " ", query_idx)
	learner.teach(
		X=X_pool[query_idx].reshape(1, -1),
		y=y_pool[query_idx].reshape(1, )
	)

	print("loop: ", _)
	performance_history.append(learner.score(train_features, train_labels))
	# remove queried instance from pool
	X_pool = np.delete(X_pool, query_idx, axis=0)
	y_pool = np.delete(y_pool, query_idx)
	# cosine_density = np.delete(cosine_density, query_idx)

print(performance_history)

plt.plot(performance_history)
plt.show()
