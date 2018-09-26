import os
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage


def parsing_annos_convert():
  """
  """
  PARSING_ANNO_OUTPUT_DIR = '/home/rantaimu/DataSpace/Dataset/Segmentation/LIP/instance-level_human_parsing/Validation/MHP_format/val/parsing_annos' # noqa
  HUMAN_LABEL_DIR         = '/home/rantaimu/DataSpace/Dataset/Segmentation/LIP/instance-level_human_parsing/Validation/Human_ids' # noqa
  GLOBAL_LABEL_DIR        = '/home/rantaimu/DataSpace/Dataset/Segmentation/LIP/instance-level_human_parsing/Validation/Category_ids' # noqa
  IMAGE_NAME_LIST         = '/home/rantaimu/DataSpace/Dataset/Segmentation/LIP/instance-level_human_parsing/Validation/val_id.txt' # noqa

  with open(IMAGE_NAME_LIST, 'r') as f:
    image_name_list = [x.strip() for x in f.readlines()]

  for image_name in tqdm(image_name_list,
                         desc='Processing {}'):
    gloabl_gt_label = np.array(PILImage.open(
      os.path.join(GLOBAL_LABEL_DIR, image_name + '.png')
    ))
    human_gt_label  = np.array(PILImage.open(
      os.path.join(HUMAN_LABEL_DIR, image_name + '.png')
    ))

    human_ids = np.unique(human_gt_label)
    bg_id_index = np.where(human_ids == 0)[0]
    human_ids = np.delete(human_ids, bg_id_index)

    for id in human_ids:
      human_part_label = (np.where(human_gt_label == id, 1, 0) * gloabl_gt_label).astype(np.uint8)
      human_part_label = PILImage.fromarray(human_part_label)
      human_part_label.save(os.path.join(PARSING_ANNO_OUTPUT_DIR,
                                         '{}_{:02d}_{:02d}.png'.format(image_name, len(human_ids), id)))


if __name__ == '__main__':
  parsing_annos_convert()
