import pycocotools
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2
import copy


def draw(polygon, image):
  if (polygon.ndim != 3):
    polygon = polygon.reshape((1, -1, 2))
  if (polygon.dtype != np.int32):
    polygon = polygon.astype(np.int32)

  return cv2.polylines(image, polygon, True, (255, 255, 255))
  # cv2.imshow('image', image)
  # cv2.waitKey(0)


def test(coco_train):
  # coco_train.info()
  # coco_train = copy.deepcopy(_coco_train)

  coco_images = coco_train.dataset['images']
  annotations = coco_train.dataset['annotations']
  categories = coco_train.dataset['categories']

  image_lables = {}

  # for categorie in categories:
  #   print(categorie)

  for image in coco_images:
    id = image['id']
    image_lables[id] = copy.deepcopy(image)

  for annotation in annotations:

    if (annotation['category_id'] != 1):
      continue

    id = annotation['image_id']
    if (image_lables.__contains__(id)):
      img = image_lables[id]
      if (img.__contains__('annotation')):
        img['annotation'].append(annotation)
      else:
        img['annotation'] = [annotation]
    else:
      print("do not contains image %d" % (id))

  filtered = {}
  ann = None
  for id, info in image_lables.items():
    if not info.__contains__('annotation'):
      continue
    if (len(info['annotation']) != 1):
      continue
    annotation = info['annotation'][0]
    print("has ann")
    if (annotation['iscrowd'] == 1):
      # RLE
      print(annotation['segmentation'])
      continue
    image = cv2.imread('/home/moriarty/Datasets/coco/train2017/' + info['file_name'])
    if (image.size == 0):
      print("image [%s] is empty" % (info['file_name']))
      continue
    ann = info
    print("has ann", ann)

    img = draw(np.array(annotation['segmentation']), image)
    cv2.imwrite("newimg.png", img)
    break
    # if (not annotation.__contains__('segmentation')):
    #   print(annotation)
    #   break
    #   continue
    filtered[id] = info
  print("contains single per image %d" % len(filtered))
  return ann

def main():

  ann_train_file='/home/moriarty/Datasets/coco/annotations/instances_train2017.json'
  coco_train = COCO(ann_train_file)
  test(coco_train)




