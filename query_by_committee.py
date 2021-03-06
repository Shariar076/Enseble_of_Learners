import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
import  pandas as pd
# loading the iris dataset
# iris = load_iris()

# visualizing the classes
# with plt.style.context('seaborn-white'):
#     plt.figure(figsize=(7, 7))
#     pca = PCA(n_components=2).fit_transform(iris['data'])
#     plt.scatter(x=pca[:, 0], y=pca[:, 1], c=iris['target'], cmap='viridis', s=5)
#     plt.title('The iris dataset')
#     plt.show()
# generate the pool

train_dataset = pd.read_table('./datasets/avila-tr.txt',header=None, sep=",")
train_data = train_dataset.values
train_features = train_data[:, :-1]
train_labels = train_data[:, -1]

lbl_names = np.unique(train_labels)
for i in range(len(lbl_names)):
    train_labels[train_labels==lbl_names[i]] = i
train_labels = train_labels.astype(int)
X_pool = deepcopy(train_features)
y_pool = deepcopy(train_labels)

print(np.shape(X_pool))
# print(np.shape(iris['target']))
# initializing Committee members
n_members = 2
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 5
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    # initializing learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_train, y_training=y_train
    )
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list)

# visualizing the initial predictions
# with plt.style.context('seaborn-white'):
#     plt.figure(figsize=(n_members*7, 7))
#     for learner_idx, learner in enumerate(committee):
#         plt.subplot(1, n_members, learner_idx + 1)
#         plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(iris['data']), cmap='viridis', s=5)
#         plt.title('Learner no. %d initial predictions' % (learner_idx + 1))
#     plt.show()

# visualizing the Committee's predictions per learner
# with plt.style.context('seaborn-white'):
#     plt.figure(figsize=(7, 7))
#     prediction = committee.predict(iris['data'])
#     plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=5)
#     plt.title('Committee initial predictions')
#     plt.show()

# query by committee
unqueried_score = committee.score(train_features, train_labels)
performance_history = [unqueried_score]
n_queries = 100
for _ in range(n_queries):
    query_idx, query_instance = committee.query(X_pool) # -> Here
    print(query_instance, " ", query_idx)
    committee.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    performance_history.append(committee.score(train_features, train_labels))
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

print(performance_history)

# visualizing the final predictions per learner
# with plt.style.context('seaborn-white'):
#     plt.figure(figsize=(n_members*7, 7))
#     for learner_idx, learner in enumerate(committee):
#         plt.subplot(1, n_members, learner_idx + 1)
#         plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(iris['data']), cmap='viridis', s=5)
#         plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
#     plt.show()
#
# # visualizing the Committee's predictions
# with plt.style.context('seaborn-white'):
#     plt.figure(figsize=(7, 7))
#     prediction = committee.predict(iris['data'])
#     plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=5)
#     plt.title('Committee predictions after %d queries' % n_queries)
#     plt.show()
plt.plot(performance_history)
plt.show()