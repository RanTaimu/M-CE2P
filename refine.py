import numpy as np
import os
from tqdm import tqdm


def extend(si, sj, instance_label, global_label, human_label, class_map):
  """
  """
  directions = [[-1, 0], [0, 1], [1, 0], [0, -1],
                [1, 1], [1, -1], [-1, 1], [-1, -1]]

  inst_class   = instance_label[si, sj]
  human_class  = human_label[si, sj]
  global_class = class_map[inst_class]
  queue = [[si, sj]]

  while len(queue) != 0:
    cur = queue[0]
    queue.pop(0)

    for direction in directions:
      ni = cur[0] + direction[0]
      nj = cur[1] + direction[1]

      if ni >= 0 and nj >= 0 and \
         ni < instance_label.shape[0] and \
         nj < instance_label.shape[1] and \
         instance_label[ni, nj] == 0 and \
         global_label[ni, nj] == global_class:
        instance_label[ni, nj] = inst_class
        # Using refined instance label to refine human label
        human_label[ni, nj] = human_class
        queue.append([ni, nj])


def refine(instance_label, human_label, global_label, class_map):
  """
  Inputs:
    [ instance_label ]
      np.array() with shape [h, w]
    [ global_label ] with shape [h, w]
      np.array()
  """
  for i in range(instance_label.shape[0]):
    for j in range(instance_label.shape[1]):
      if instance_label[i, j] != 0:
        extend(i, j, instance_label, global_label, human_label, class_map)
