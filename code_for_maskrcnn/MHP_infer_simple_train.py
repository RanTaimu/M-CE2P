#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np


from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# data_root      = "/home/rantaimu/Documents/Dataset/Segmentation/LIP/instance-level_human_parsing"
# src_img_folder = "Images"
# src_img_sub_folder = 'test_images'
# src_gt_folder  = "Categories"
# src_gt_sub_folder = 'val_segmentations'
# dst_img_sub_folder = 'val_images_sub'
# dst_gt_sub_folder = 'val_segmentations_sub'
# src_img_list_file = 'mul_test_id.txt'
# dst_subimg_list_file = 'test_sub_image_id.txt'

data_root           = "/home/rantaimu/Documents/Dataset/Segmentation/MHP/LV-MHP-v2"
task                = os.path.join(data_root, "train")
src_img_folder      = os.path.join(task, "images")
src_cat_gt_folder   = os.path.join(task, "parsing_annos")
src_img_list_file   = os.path.join(data_root, 'list', 'train.txt')  # os.path.join(task, 'tttt.txt')
# src_human_gt_folder = os.path.join(task, 'Human_ids')
dst_img_folder      = os.path.join(task, 'ImagesSub')
dst_cat_gt_folder   = os.path.join(task, 'CategoriesSub')
dst_img_list_file   = os.path.join(task, 'train_sub.txt')
dst_whole_gt_folder = os.path.join(task, 'AnnoWhole')
dst_box_list_folder = os.path.join(task, 'Boxes')

visualize = False

if not os.path.exists(dst_img_folder):
    os.makedirs(dst_img_folder)

if not os.path.exists(dst_cat_gt_folder):
    os.makedirs(dst_cat_gt_folder)

if not os.path.exists(dst_whole_gt_folder):
    os.makedirs(dst_whole_gt_folder)

if not os.path.exists(dst_box_list_folder):
    os.makedirs(dst_box_list_folder)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='models/coco_2014_train_coco_2014_valminusminival_model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='./results',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def gt_process(gt_folder, name_prefix):
    """
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
        single_human_gt = cv2.imread(os.path.join(gt_folder, name))[:, :, 2]
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
    # cv2.imwrite("hehehehe.png", cat_gt)
    # cv2.imwrite("hahahahaa.png", human_gt)
    # exit()
    return cat_gt, human_gt


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    # Get image name list.
    if visualize:
        if os.path.isdir(args.im_or_folder):
            im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
        else:
            im_list = [args.im_or_folder]
    else:
        with open(src_img_list_file, 'r') as f:
            im_list = f.readlines()
            for i, im_name in enumerate(im_list):
                im_list[i] = im_name.strip()

    f = open(dst_img_list_file, "w")
    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))

        if visualize:
            im_full_name = im_name
            im = cv2.imread(im_full_name, cv2.IMREAD_COLOR)
        else:
            # Loading picture
            im_full_name = os.path.join(src_img_folder, im_name + '.jpg')
            if not os.path.exists(im_full_name):
                print('%s: No such file.' % im_full_name)
                exit()
            im = cv2.imread(im_full_name, cv2.IMREAD_COLOR)

            # Loading groundtruth
            cat_gt, human_gt = gt_process(src_cat_gt_folder, im_name)

            # cv2.imwrite(os.path.join(dst_whole_gt_folder, im_name + '.png'),
            #             cat_gt)
            # continue

            # cat_gt_full_name = os.path.join(src_cat_gt_folder, im_name + '.png')
            # if not os.path.exists(cat_gt_full_name):
            #     print('%s: No such file.' % cat_gt_full_name)
            #     exit()
            # cat_gt = cv2.imread(cat_gt_full_name, cv2.IMREAD_GRAYSCALE)

            # human_gt_full_name = os.path.join(src_human_gt_folder, im_name + '.png')
            # if not os.path.exists(human_gt_full_name):
            #     print('%s: No such file.' % human_gt_full_name)
            #     exit()
            # human_gt = cv2.imread(human_gt_full_name, cv2.IMREAD_GRAYSCALE)

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        boxes, masks, classes = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            visualize=visualize
        )
        if not visualize and boxes is not None:
            get_objs(
                im, cat_gt, human_gt, im_name,
                boxes, masks, classes, dummy_coco_dataset, f
            )

    f.close()


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(np.reshape(index, index.shape[0] * index.shape[1]))
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def get_IoU(cmat):
    """
    """
    return cmat[1, 1] / (cmat[1, 1] + cmat[1, 0] + cmat[0, 1])


def get_objs(im, cat_gt, human_gt, im_id,
                   boxes, masks, classes, dataset, list_file, thresh=0.7):
    """
    """
    im_h, im_w = im.shape[0], im.shape[1]
    scale = 1.2
    comfirmed_boxes = []
    for i in xrange(boxes.shape[0]):
        box = boxes[i]
        if(box[4] < thresh):
            continue

        if dataset.classes[classes[i]] != 'person':
            continue

        # Judge the mean IoU to decide whether the box should be kept.
        mask = masks[:, :, i]
        human_gt_labels = list(np.unique(human_gt))
        ok = False
        for gt_label in human_gt_labels:
            if gt_label == 0:
                continue
            IoU = get_IoU(
                get_confusion_matrix(
                    (human_gt == gt_label).astype('int32'),
                    mask.astype('int32'),
                    2
                ))
            if IoU >= 0.5:
                ok = True
                break

        if not ok:
            continue

        bw = box[2] - box[0]
        bh = box[3] - box[1]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        box[0] = max(0, cx - bw * scale / 2)
        box[1] = max(0, cy - bh * scale / 2)
        box[2] = min(im_w, cx + bw * scale / 2)
        box[3] = min(im_h, cy + bh * scale / 2)

        comfirmed_boxes.append(box)

    # crop and save
    box_file = open(os.path.join(dst_box_list_folder, im_id + '.txt'), 'w')
    for i, box in enumerate(comfirmed_boxes):
        res_name = '%07d_%02d' % (int(im_id), i)
        list_file.write(res_name + '\n')
        cv2.imwrite(os.path.join(
            dst_img_folder,
            res_name + '.jpg'),
                    im[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        )
        cv2.imwrite(os.path.join(
            dst_cat_gt_folder,
            res_name + '.png'),
                    cat_gt[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        )
        box_file.write(str(int(box[0])) + ' ')
        box_file.write(str(int(box[1])) + ' ')
        box_file.write(str(int(box[2])) + ' ')
        box_file.write(str(int(box[3])) + '\n')
    box_file.close()



if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=-1'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
