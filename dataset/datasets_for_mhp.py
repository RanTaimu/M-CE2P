import os
import os.path as osp
import numpy as np
import cv2
from torch.utils import data


class CIHPDataTestSet(data.Dataset):
  """
  """
  def __init__(self,
               root, list_path, box_dir, img_ext='jpg',
               crop_size=(473, 473), mean=(128, 128, 128)):
    self.root      = root
    self.list_path = list_path
    self.crop_size = crop_size
    self.mean      = mean
    self.box_dir   = box_dir

    self.img_ids = [i_id.strip().split()[0] for i_id in open(list_path)]

    self.files = []
    for img_id in self.img_ids:
      fbox_path = osp.join(box_dir, img_id + '.txt')

      item = {}
      name     = osp.splitext(osp.basename(img_id))[0]
      img_file = osp.join(self.root, img_id + img_ext)

      boxes = None
      if os.path.exists(fbox_path):
        boxes = []
        with open(fbox_path, 'r') as box_file:
          for box_cood in box_file.readlines():
            boxes.append(box_cood.strip().split(' '))
      # else:
      #   print('{} has no boxes, take whole image as an instance'.format(img_id))

      item["img"]   = img_file
      item["name"]  = name
      item["boxes"] = boxes
      self.files.append(item)

  def __len__(self):
    return len(self.files)

  def generate_scale_image(self, image, f_scale):
    image = cv2.resize(image, None, fx=f_scale, fy=f_scale,
                       interpolation=cv2.INTER_LINEAR)
    return image

  def __getitem__(self, index):
    datafiles = self.files[index]
    image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
    size  = image.shape

    if datafiles["boxes"] is None:
      datafiles["boxes"] = [[0, 0, size[1], size[0]]]

    image = np.asarray(image, np.float32)
    image -= self.mean

    sub_images = []
    for box in datafiles["boxes"]:
      sub_image = image[int(box[1]):int(box[3]),
                        int(box[0]):int(box[2]),
                        :]
      sub_image = cv2.resize(sub_image, dsize=self.crop_size,
                             interpolation=cv2.INTER_LINEAR)
      sub_images.append(sub_image.transpose((2, 0, 1)).copy())

    image = cv2.resize(image, dsize=self.crop_size,
                       interpolation=cv2.INTER_LINEAR)
    image = image.transpose((2, 0, 1)).copy()
    name = datafiles["name"]

    return image, name, np.array(size), sub_images, datafiles["boxes"]
