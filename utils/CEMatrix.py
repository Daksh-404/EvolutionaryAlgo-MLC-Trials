import numpy as np
from scipy import stats

def joint_entropy(first,second):
  probs=[]
  for value1 in set(first):
    for value2 in set(second):
      probs.append(np.mean(np.logical_and(first == value1, second == value2)))
  return stats.entropy(probs)

def entropy(X):
    probs = [np.mean(X == c) for c in set(X)]
    return stats.entropy(probs)

def construct_conditional_entropy_matrix(label_space):
    labels = label_space.shape[-1]
    ans = np.zeros(shape = (labels, labels))
    for i in range(labels):
      for j in range(labels):
        if i != j:
          ans[i][j] = joint_entropy(label_space[:, i], label_space[:, j]) - entropy(label_space[:, i])
    return ans