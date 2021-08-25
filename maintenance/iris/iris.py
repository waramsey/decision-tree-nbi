import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load iris 
iris = load_iris()
X = iris.data
y = iris.target
X_train,  X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf.fit(X_train, y_train)

# Tree Structure
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold

# Creating a dictionary
featureDict = defaultdict()
for index, feature_name in enumerate(iris.feature_names):
    featureDict[index] = feature_name

print("\nFeature Dict:", featureDict)

# Printing left child and printing right child
counter = 0
for leftChild, rightChild in zip(children_left, children_right):
    print('\nCounter:', counter)
    print('\nLeft child:', leftChild)
    print('\nRight child:', rightChild)
    counter = counter + 1
    print('--'*4)

# function to identify leave nodes and split nodes
# Tree Structure
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# Start with the root node
stack = [[0, 0]] #[(node_id, depth)] 

while len(stack) > 0:
    node_id, depth = stack.pop()

    # if the left and right child of a node is not the same we have a split
    is_split_node = children_left[node_id] != children_right[node_id]

    # If a split node, append left and right children and depth to 'stack'
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print("\n Printing leaves and split nodes ")
print(is_leaves)

#print(is_split_node)

for i in range(n_nodes):
    if is_leaves[i]:
        print("{space} node={node} is a leaf node.".format(space=node_depth[i] *"\t", node=i))
    else:
        print("{space} node is a split-node: "
              "go to node {left} if X[:, {feature}] <= {threshold} "
              "else to node {right}.".format(
              space=node_depth[i]*"\t",
              node=i,
              left=children_left[i],
              feature=featureDict[feature[i]],
              threshold=threshold[i],
              right=children_right[i]))

# Decision path
node_indicator = clf.decision_path(X_test)
leaf_id = clf.apply(X_test)


# For every data dample show the decision path
sample_id = 10
# Obtain ids of the nodes 'sample id' goes through i.e., row `sample_id`
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

# Gather all the nodes features split nodes, and their children  
for node_id in node_index:
    # Continue to the next node if it a leaf node
    if leaf_id[sample_id] == node_id:
        continue

    # check if value of the split feature for sample 0 is below threshold
    if (X_test[sample_id,  feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"
    print("decision node {node}:(X_test[{sample}, {feature}] = {value})" " {inequality} {threshold}".format(
                                                                                    node=node_id,
                                                                                    sample=sample_id,
                                                                                    feature=feature[node_id],
                                                                                    value=X_test[sample_id, feature[node_id]],
                                                                                    inequality=threshold_sign,
                                                                                    threshold=threshold[node_id]))
#node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
#is_leaves = np.zeros(shape=n_nodes, dtype=bool)
#stack=[(0,0
