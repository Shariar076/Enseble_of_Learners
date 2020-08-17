import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
import pandas as pd


train_dataset = pd.read_table('./datasets/avila-tr.txt',header=None, sep=",")
train_data = train_dataset.values
train_features = train_data[:, :-1]
train_labels = train_data[:, -1]
# np.savetxt('out.txt',train_features)


# train_data = load_breast_cancer()
# train_features = train_data['data']
# train_labels = train_data['target']

#only for manual datasets
lbl_names = np.unique(train_labels)
for i in range(len(lbl_names)):
    train_labels[train_labels==lbl_names[i]] = i
train_labels = train_labels.astype(int)


X_pool = deepcopy(train_features)
y_pool = deepcopy(train_labels)

n_members = 5
learner_list = list()
for member_idx in range(n_members):
    # initial training data
    n_initial = len(np.unique(y_pool))
    # train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    train_idx = np.unique(y_pool,return_index=True)[1] #instead of takin g randomly take one sample of each class IMP!!!

    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]
    # print(y_train,train_idx)

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

# query by committee
unqueried_score = committee.score(train_features, train_labels)
performance_history = [unqueried_score]
n_queries = 200
for _ in range(n_queries):
    query_idx, query_instance = committee.query(X_pool) # -> Here
    # print(query_instance, " ", query_idx)
    committee.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    # print("loop: ", _)
    performance_history.append(committee.score(train_features, train_labels))
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

print(performance_history)

plt.plot(performance_history)
plt.show()