import os
import numpy as np
from metrics import GlobalMetrics, InstanceMetrics


# DATA_ROOT              = '/home/rantaimu/DataSpace/Dataset/Segmentation/LIP/instance-level_human_parsing/Validation' # noqa
DATA_ROOT              = '/home/rantaimu/DataSpace/Dataset/Segmentation/MHP/LV-MHP-v2/val/LIP_format' # noqa
IMAGE_LIST             = os.path.join(DATA_ROOT, 'val_id.txt')
GLOBAL_GT_FOLDER       = os.path.join(DATA_ROOT, 'Category_ids')
INSTANCE_GT_FOLDER     = os.path.join(DATA_ROOT, 'Instance_ids')

# RESULT_ROOT            = '/home/rantaimu/Documents/Program/ImageSegmentation/ExperimentalData/LIP_with_refine' # noqa
# RESULT_TYPE            = [
#   'LIP-MRCNN-gt-whole',
#   'LIP-MRCNN-whole',
#   'LIP-MRCNN-gt',
#   'LIP-gt-whole',
#   'LIP-MRCNN',
#   'LIP-gt',
#   'LIP-whole'
# ]

RESULT_ROOT            = '/home/rantaimu/Documents/Program/ImageSegmentation/ExperimentalData/MHP_with_refine' # noqa
RESULT_TYPE            = [
  'MHP-MRCNN-gt-whole',
  'MHP-MRCNN-whole',
  'MHP-MRCNN-gt',
  'MHP-gt-whole',
  'MHP-MRCNN',
  'MHP-gt',
  'MHP-whole'
]

NUM_CLASS = 59


def get_paths(global_result_folder, global_gt_folder):
  """
  Get the list of result images and groundtruth images.
  """
  with open(IMAGE_LIST, 'r') as f:
    file_names = [file_name.strip() for file_name in f.readlines()]

  global_result_paths = [
    os.path.join(global_result_folder, fn + '.png') for fn in file_names
  ]
  global_gt_paths     = [
    os.path.join(global_gt_folder, fn + '.png') for fn in file_names
  ]

  return global_result_paths, global_gt_paths


def main():
  """
  """

  for result_type in RESULT_TYPE:
    print("{} results:".format(result_type))
    global_result_folder = os.path.join(RESULT_ROOT, result_type, 'global_seg')
    global_gt_folder     = GLOBAL_GT_FOLDER
    global_result_paths, global_gt_paths = get_paths(
      global_result_folder,
      global_gt_folder
    )
    global_metrics   = GlobalMetrics(global_result_paths,
                                     global_gt_paths,
                                     NUM_CLASS)
    instance_metrics = InstanceMetrics(
      os.path.join(RESULT_ROOT, result_type, 'instance_parsing'),
      INSTANCE_GT_FOLDER,
      NUM_CLASS
    )

    print('Pixel Accuracy: {:.4f}'.format(
      global_metrics.get_pixel_accuray()
    ))
    print('Mean Pixel Accuracy: {:.4f}'.format(
      global_metrics.get_mean_pixel_accuracy()
    ))
    print('Mean IoU: {:.4f}'.format(
      global_metrics.get_mean_IoU()
    ))
    print('Frequency Weighted IoU: {:.4f}'.format(
      global_metrics.get_frequency_weighted_IoU()
    ))

    AP_map = instance_metrics.compute_AP()
    for thre in AP_map.keys():
      print('threshold: {:.2f}, AP^r: {:.4f}'.format(thre, AP_map[thre]))
    print('Mean AP^r: {}'.format(
      np.nanmean(np.array(list(AP_map.values())))
    ))
    print('=' * 80)


if __name__ == '__main__':
  main()
