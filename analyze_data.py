import os.path

import pycocotools
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2
import copy
import time
import csv
import torch

class Sample:
  def __init__(self, data:list):
    #@data: [id, file_name, height, width, area, [segmentation]]
    self.data = data
  def index(self):
    return self.data[0]
  def file_name(self):
    return self.data[1]
  def image_size(self):
    #width, height
    return (self.data[3], self.data[2])
  def segment_area(self):
    return self.data[4]
  def segmentation(self):
    return self.data[5:]


def draw(polygon, image, color=(0, 0, 255)):
  if isinstance(polygon, list):
    polygon = np.array(polygon)
  if (polygon.ndim != 3):
    polygon = polygon.reshape((1, -1, 2))
  if (polygon.dtype != np.int32):
    polygon = polygon.astype(np.int32)

  return cv2.polylines(image, polygon, True, color, thickness=3)
  # cv2.imshow('image', image)
  # cv2.waitKey(0)

def fillPoly(polygon, image, color=(1, 1, 1)):
  if isinstance(polygon, list):
    polygon = np.array([polygon], dtype=np.int32)
  if (polygon.ndim != 3):
    polygon = polygon.reshape((1, -1, 2))
  if (polygon.dtype != np.int32):
    polygon = polygon.astype(np.int32)
  image = cv2.fillPoly(image, polygon, color)
  return image


class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset:list, path_to_image:str):
    self.path = copy.deepcopy(path_to_image)
    self.dataset = {}
    self.size = len(dataset)
    for idx, sample in enumerate(dataset):
      self.dataset[idx] = sample
  def __len__(self):
    return self.size
  def __getitem__(self, idx):
    item = Sample(self.dataset[idx])
    path = os.path.join(self.path, item.file_name())
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    lable = np.zeros(image.shape[0:2], dtype=np.uint8)
    fillPoly(item.segmentation(), lable, color=(1.0, 1.0, 1.0))

    image = (image.astype(dtype=np.float32) / 255.0 * 2) - 1.0
    lable = lable.astype(dtype=np.float32)

    return [image, lable]

def showImages(dataset, src = '/home/moriarty/Datasets/coco/train2017', dst = '/home/moriarty/Datasets/coco/draw'):
  '''
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  for sample in dataset:
    img_path = os.path.join(src, sample[1])
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = draw(sample[5:], image)
    img_path = os.path.join(dst, sample[1])

    cv2.circle(image, np.array(sample[5:7], dtype=np.int32), 5, (255, 0, 0), thickness=-1)
    print("draw circle position %s" %str(sample[5:7]))

    cv2.imwrite(img_path, image)

def filter(dataset:list):
  '''
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  ndataset = []
  area_threshold = 0.1
  edge = 10
  for sample in dataset:
    #area ratio
    img_area = sample[2] * sample[3]
    ratio = sample[4] / img_area
    if (ratio < area_threshold):
      continue

    #do not too close to edge of image
    segmentation = np.array(sample[5:]).reshape(-1, 2)
    num_poly = segmentation.shape[0]
    invalid = (segmentation[:, 0] < edge) | (segmentation[:, 0] > (sample[3] - edge)) | (segmentation[:, 1] < edge) | (segmentation[:, 0] > (sample[2] - edge))
    if (np.sum(invalid) > 2):
      continue
    ndataset.append(sample)

  return ndataset


def saveDataset(dataset:list, path:str):
  '''
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''

  file = open(path, 'w')
  writer = csv.writer(file)
  for sample in dataset:
    writer.writerow(sample)
  file.close()

def loadDataset(path:str):
  '''
  @return:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  print("Loading Dataset from %s" %path)
  begin = time.time()
  file = open(path, 'r')
  reader = csv.reader(file)
  dataset = []
  for row in reader:
    sample = [int(row[0]), row[1], int(row[2]), int(row[3])]
    sample.extend([float(_) for _ in row[4:]])
    dataset.append(sample)
    # print("sample ", sample)
  file.close()
  cost = time.time() - begin
  print("Load dataset cost time %f sec" %cost)
  return dataset

def translateCocoDict(datas:dict):
  '''
  @datas: a dict, {id(int), infos}
  @infos: a dict, has keys ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id', 'annotation']
  @annotation: a list, typically contains only one object
  @annotation[0]: a dict, has keys ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
  @annotation[0][segmentation]: [[x0, y0, x1, y1, ..., ]]

  @return: [[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  begin = time.time()
  print("Translating Coco Datasets...")
  translated = []
  for id, infos in datas.items():
    annotation = infos['annotation'][0]
    info = [id, infos['file_name'], infos['height'], infos['width'], annotation['area']]
    info.extend(annotation['segmentation'][0])
    translated.append(info)
  print("Translate finish, cost %f sec" %(time.time() - begin))
  return translated

def test(coco_train):
  begin = time.time()
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
  size = len(image_lables)
  current = 0
  for id, info in image_lables.items():
    current+=1
    if not info.__contains__('annotation'):
      continue
    if (len(info['annotation']) != 1):
      continue
    annotation = info['annotation'][0]
    # print("has ann")
    if (annotation['iscrowd'] == 1):
      # RLE
      print(annotation['segmentation'])
      continue
    image = cv2.imread('/home/moriarty/Datasets/coco/train2017/' + info['file_name'])
    if (image.size == 0):
      print("image [%s] is empty" % (info['file_name']))
      continue
    # print("has ann %d, %s" %(id, str(info)))
    print("preprocessed data %d/%d, %.2f" %(current, size, 100.0 * current/size))

    # img = draw(np.array(annotation['segmentation']), image)
    # cv2.imwrite("newimg.png", img)
    # break
    # if (not annotation.__contains__('segmentation')):
    #   print(annotation)
    #   break
    #   continue
    filtered[id] = info
  print("contains single per image %d" % len(filtered))
  print("return filtered type ", type(filtered))
  return filtered

def main():
  ann_train_file='/home/moriarty/Datasets/coco/annotations/instances_train2017.json'
  print("coco train file reading")
  coco_train = COCO(ann_train_file)
  print("coco train file finish")
  return test(coco_train)
# main()




