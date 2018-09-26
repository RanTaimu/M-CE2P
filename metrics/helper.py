import os
from PIL import Image as PILImage
import numpy as np
from tqdm import tqdm


def generate_help_file(data_root):
  """
  Generate help file for groundtruth to accelerate the computation of AP.
  """

  DATA_ROOT          = data_root # = '/home/rantaimu/DataSpace/Dataset/Segmentation/LIP/instance-level_human_parsing/Validation' # noqa
  GLOBAL_GT_FOLDER   = os.path.join(DATA_ROOT, 'Category_ids')
  INSTANCE_GT_FOLDER = os.path.join(DATA_ROOT, 'Instance_ids')
  HUMAN_GT_FOLDER    = os.path.join(DATA_ROOT, 'Human_ids')

  IMAGE_NAME_LIST    = os.path.join(DATA_ROOT, 'val.txt')

  with open(IMAGE_NAME_LIST, 'r') as f:
    image_name_list = [x.strip() for x in f.readlines()]

  pbar = tqdm(total=len(image_name_list), desc="Generating help file.")
  for count, image_name in enumerate(image_name_list):
    # print('{} / {}: {}'.format(count + 1, len(image_name_list), image_name))
    global_gt_img = PILImage.open(
      os.path.join(GLOBAL_GT_FOLDER, image_name + '.png')
    )
    human_gt_img  = PILImage.open(
      os.path.join(HUMAN_GT_FOLDER, image_name + '.png')
    )
    instance_gt_img = PILImage.open(
      os.path.join(INSTANCE_GT_FOLDER, image_name + '.png')
    )
    global_gt_img   = np.array(global_gt_img)
    human_gt_img    = np.array(human_gt_img)
    instance_gt_img = np.array(instance_gt_img)
    assert(global_gt_img.shape == human_gt_img.shape and
           global_gt_img.shape == instance_gt_img.shape)

    acce_f = open(os.path.join(INSTANCE_GT_FOLDER, image_name + '.txt'), 'w')

    instance_ids = np.unique(instance_gt_img)
    background_id_index = np.where(instance_ids == 0)[0]
    instance_ids = np.delete(instance_ids, background_id_index)

    for inst_id in instance_ids:
      inst_id_index = np.where(instance_gt_img == inst_id)
      human_ids = human_gt_img[inst_id_index]
      human_ids = np.unique(human_ids)
      assert(human_ids.shape[0] == 1)

      part_class_ids = global_gt_img[inst_id_index]
      part_class_ids = np.unique(part_class_ids)
      assert(part_class_ids.shape[0] == 1)

      acce_f.write('{} {} {}\n'.format(int(inst_id),
                                       int(part_class_ids[0]),
                                       int(human_ids[0])))
    acce_f.close()
    pbar.update(1)
  pbar.close()

if __name__ == '__main__':
  """
  """
  generate_help_file()
