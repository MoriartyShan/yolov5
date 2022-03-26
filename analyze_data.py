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
import models.common as mc

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
  def create_label(self, _new_size, root):
    path = os.path.join(root, self.file_name())
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    old_size = np.array(self.image_size())
    new_size = np.array(_new_size)

    segmentation = np.array(self.segmentation()).reshape(-1, 2)

    #resize
    scale = new_size / old_size
    if (scale[0] < scale[1]):
      scale[1] = scale[0]
    else:
      scale[0] = scale[1]
    size = (scale[0] * old_size.astype(scale.dtype)).astype(np.int32)

    segmentation *= scale[0]
    print("resize image from, ", old_size, "=>", size, image.shape, self.file_name())
    _image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)

    current_center = size / 2
    to_center = new_size / 2
    translate = (to_center - current_center).astype(np.int32)
    print("current_center,", current_center, ",to_center,", to_center, 'translate', translate)

    segmentation += translate

    image = np.zeros((_new_size[1], _new_size[0], 3), dtype=np.float32)
    print('_image.shape ', _image.shape, image.shape)
    image[translate[1]:(translate[1] + _image.shape[0]), translate[0]:(translate[0] + _image.shape[1]), :] = _image

    label = np.zeros(image.shape[0:2], dtype=np.uint8)
    fillPoly(segmentation, label, color=(1.0, 1.0, 1.0))

    # _image = image.copy()
    # fillPoly(segmentation, _image, color=(1.0, 1.0, 1.0))

    image = (image.astype(dtype=np.float32).transpose(2, 0, 1) / (255.0 / 2)) - 1.0
    return image, label, _image


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
    self.image_size = (640, 640)
    self.path = copy.deepcopy(path_to_image)
    self.dataset = {}
    self.size = len(dataset)
    for idx, sample in enumerate(dataset):
      self.dataset[idx] = sample
  def __len__(self):
    return self.size
  def __getitem__(self, idx):
    item = Sample(self.dataset[idx])

    image, label = item.create_label(self.image_size, self.path)


    return [image, label]

def showImages(dataset, src = '/home/moriarty/Datasets/coco/train2017', dst = '/home/moriarty/Datasets/coco/draw'):
  '''
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  for sample in dataset:
    sample = Sample(sample)
    img_path = os.path.join(src, sample.file_name())
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = draw(sample.segmentation(), image)
    img_path = os.path.join(dst, sample.file_name())

    cv2.circle(image, np.array(sample.segmentation()[0:2], dtype=np.int32), 5, (255, 0, 0), thickness=-1)
    print("draw circle position %s" %str(sample.segmentation()[0:2]))

    cv2.imwrite(img_path, image)

def filter(dataset:list):
  '''
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  ndataset = []
  area_threshold = 0.1
  edge = 10
  for _ in dataset:
    #area ratio
    sample = Sample(_)
    shape = sample.image_size()
    img_area = shape[0] * shape[1]
    ratio = sample.segment_area() / img_area
    if (ratio < area_threshold):
      continue

    #do not too close to edge of image
    segmentation = np.array(sample.segmentation()).reshape(-1, 2)
    invalid = (segmentation[:, 0] < edge) | (segmentation[:, 0] > (shape[0] - edge)) | (segmentation[:, 1] < edge) | (segmentation[:, 0] > (shape[1] - edge))
    if (np.sum(invalid) > 2):
      continue
    ndataset.append(_)

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

def train(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader, optimizer,
          compute_loss, epoch, device):
  size = len(data_loader) / data_loader.batch_size
  show_count = (size - 1) // 10
  show_count = 1 if (show_count == 0) else show_count
  show_count = 100 if (show_count > 100) else show_count
  total_loss = 0.0

  model.train()
  epoch_begin = time.time()
  for index, batch_data in enumerate(data_loader):
    image = batch_data[0].to(device)
    label = batch_data[1].to(device)

    optimizer.zero_grad()
    output = model(image)
    loss = compute_loss(output, label)
    total_loss += loss.data

    loss.backward()
    optimizer.step()
    if ((index % show_count) == 0):
      cost = time.time() - epoch_begin
      print('train %d in ep %d %0.2f%%, %f, %f, %f sec, left %f sec' % (
        index, epoch, 100.0 * index / size, total_loss / index, loss.data,
        cost, cost / index * (size - index)))
  cost = time.time() - epoch_begin
  print('epoch %d finished, trained %d samples, average loss %f, cost time %f sec' % (
    epoch, size, total_loss / index, cost))
  return total_loss / size, cost



class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.channels = [3, 8, 16, 32, 1]
    self.kernels = [7, 5, 3, 3]
    convs = []
    num = len(self.kernels)
    for i in range(num):
      convs.append(mc.Conv(self.channels[i], self.channels[i+1], self.kernels[i]))
    self.conv = torch.nn.Sequential(*convs)
  def forward(self, x):
    return self.conv(x)


def main():
  default_type = torch.float32
  device = torch.device("cuda:0")
  dataset_path = '/home/moriarty/Projects/yolov5/x.csv'
  image_path = '/home/moriarty/Datasets/coco/train2017'
  dataset = loadDataset(dataset_path)
  dataset = filter(dataset)
  dataset = Dataset(dataset, image_path)
  size = len(dataset)
  train_size = int(0.9 * size)

  model = Model()
  model = model.to(device)
  compute_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

  trset, teset = torch.utils.data.random_split(
    dataset,
    [train_size, size - train_size],
    generator=torch.Generator().manual_seed(42))

  trloader = torch.utils.data.DataLoader(
    trset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    prefetch_factor=2,
    persistent_workers=False)

  teloader = torch.utils.data.DataLoader(
    teset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    prefetch_factor=2,
    persistent_workers=False)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                               betas=(0.9, 0.99))

  for i in range(100):
    train(model, trloader, optimizer, compute_loss, i, device)

def load_dataset():
  dataset = loadDataset('x.csv')
  dataset = filter(dataset)
  return dataset


def test_origin():
  ann_train_file='/home/moriarty/Datasets/coco/annotations/instances_train2017.json'
  print("coco train file reading")
  coco_train = COCO(ann_train_file)
  print("coco train file finish")
  return test(coco_train)
# main()




