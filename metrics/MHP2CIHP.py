import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2

from helper import generate_help_file


DATA_ROOT = '/home/rantaimu/DataSpace/Dataset/Segmentation/MHP/LV-MHP-v2/val'
SRC_CAT_GT_DIR   = osp.join(DATA_ROOT, "parsing_annos")
IMAGE_NAME_LIST  = osp.join(DATA_ROOT, "val.txt")

OUTPUT_ROOT = osp.join(DATA_ROOT, 'LIP_format')
DST_CAT_GT_DIR   = osp.join(OUTPUT_ROOT, 'Category_ids')
DST_INST_GT_DIR  = osp.join(OUTPUT_ROOT, 'Instance_ids')
DST_HUMAN_GT_DIR = osp.join(OUTPUT_ROOT, 'Human_ids')

if not osp.exists(DST_HUMAN_GT_DIR):
  os.makedirs(DST_HUMAN_GT_DIR)
if not osp.exists(DST_CAT_GT_DIR):
  os.makedirs(DST_CAT_GT_DIR)
if not osp.exists(DST_INST_GT_DIR):
  os.makedirs(DST_INST_GT_DIR)

def gt_process(gt_folder, name_prefix):
  """
  Get Category_ids and Human_ids
  """
  human_num = 1
  while not os.path.exists(
      os.path.join(gt_folder,
                   '%s_%02d_01.png' % (name_prefix, human_num))
  ):
    human_num += 1

  cat_gt   = None
  human_gt = None
  for human_id in range(1, human_num + 1):
    # Label is only put in R channel.
    name = '%s_%02d_%02d.png' % (name_prefix, human_num, human_id)
    single_human_gt = cv2.imread(osp.join(gt_folder, name))[:, :, 2]
    original_shape = single_human_gt.shape
    single_human_gt = single_human_gt.reshape(-1)

    if cat_gt is None:
      cat_gt = np.zeros_like(single_human_gt)
    if human_gt is None:
      human_gt = np.zeros_like(single_human_gt)

    indexes           = np.where(single_human_gt != 0)
    cat_gt[indexes]   = single_human_gt[indexes]
    human_gt[indexes] = human_id

  assert(cat_gt.max() <= 58)
  assert(human_gt.max() <= 58)

  cat_gt = cat_gt.reshape(original_shape)
  human_gt = human_gt.reshape(original_shape)
  return cat_gt, human_gt


def get_instance(cat_gt, human_gt):
  """
  """
  instance_gt = np.zeros_like(cat_gt, dtype=np.uint8)

  human_ids = np.unique(human_gt)
  bg_id_index = np.where(human_ids == 0)[0]
  human_ids = np.delete(human_ids, bg_id_index)

  total_part_num = 0
  for id in human_ids:
    human_part_label = (np.where(human_gt == id, 1, 0) * cat_gt).astype(np.uint8)
    part_classes = np.unique(human_part_label)

    for part_id in part_classes:
      if part_id == 0:
        continue
      total_part_num += 1
      instance_gt[np.where(human_part_label == part_id)] = total_part_num

  if total_part_num >= 255:
    print("total_part_num exceed: {}".format(total_part_num))
    exit()

  return instance_gt


def do_work():
  """
  """
  with open(IMAGE_NAME_LIST, 'r') as f:
    image_name_list = [x.strip() for x in f.readlines()]

  for image_name in tqdm(image_name_list,
                         desc="Processing"):
    cat_gt, human_gt = gt_process(SRC_CAT_GT_DIR, image_name)
    instance_gt      = get_instance(cat_gt, human_gt)

    cv2.imwrite(osp.join(DST_CAT_GT_DIR, image_name + '.png'),
                cat_gt)
    cv2.imwrite(osp.join(DST_HUMAN_GT_DIR, image_name + '.png'),
                human_gt)
    cv2.imwrite(osp.join(DST_INST_GT_DIR, image_name + '.png'),
                instance_gt)

  generate_help_file(OUTPUT_ROOT)


if __name__ == '__main__':
  do_work()
