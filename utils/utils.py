import numpy as np
from scipy import stats
from tqdm import tqdm

def joint_entropy(first, second, uniq_first, uniq_second):
  probs=[]
  for value1 in uniq_first:
    for value2 in uniq_second:
      probs.append(np.mean(np.logical_and(first == value1, second == value2)))
  return stats.entropy(probs)

def entropy(X, unique_x):
    probs = [np.mean(X == c) for c in unique_x]
    return stats.entropy(probs)

def construct_conditional_entropy_matrix(label_space):
    labels = label_space.shape[-1]
    ans = np.zeros(shape = (labels, labels))
    idx_to_unique_vals = {}
    for i in range(labels):
       idx_to_unique_vals[i] = set(label_space[:, i])

    for i in tqdm(range(labels)):
      col_i = label_space[:, i]
      unique_values_i = idx_to_unique_vals[i]
      for j in range(labels):
        if i != j:
          ans[i][j] = joint_entropy(col_i, label_space[:, j], unique_values_i, idx_to_unique_vals[j]) - entropy(col_i, unique_values_i)
    return ans

#  BELOW FUNCTION WAS ADDED RECENTLY
def construct_mutual_information_matrix(label_space):
    labels = label_space.shape[-1]
    ans = np.zeros(shape = (labels, labels))
    idx_to_unique_vals = {}
    for i in range(labels):
      idx_to_unique_vals[i] = set(label_space[:, i])

    for i in tqdm(range(labels)):
      col_i = label_space[:, i]
      unique_values_i = idx_to_unique_vals[i]
      for j in range(labels):
        if i != j:
          col_j = label_space[:, j]
          ans[i][j] = joint_entropy(col_i, col_j, unique_values_i, idx_to_unique_vals[j]) - entropy(col_i, unique_values_i)
          ans[i][j] = entropy(col_j, idx_to_unique_vals[j]) - ans[i][j]  
    return ans