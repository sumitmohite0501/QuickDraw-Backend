from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image

import config


class dataloader(Dataset):

  def __init__(self, splitType='train'):

    self.splitType = splitType
    self.allClasses = sorted(os.listdir(config.basePath[self.splitType]))
    print(self.allClasses)
    self.allData = {}
    self.allImages = []
    self.allTargets = []

    for no, imgClass in enumerate(self.allClasses):
      self.allData[imgClass] = []
      imgPath = sorted(os.listdir(config.basePath[self.splitType] + '/' + imgClass + '/image/'))
      self.allImages += imgPath
      self.allTargets += [no]*len(imgPath)
      # for image in imgPath:
      #   image = self.process(config.basePath[self.splitType] + '/' + imgClass + '/image/' + image)
      #
      #   self.allData[imgClass].append(image)

  def process(self, path):

    # image = Image.fromarray(plt.imread(path)[:, :, 3])
    # image = Image.open(path).convert('RGB')
    # print(image.size)
    image = Image.open(path)
    arr = np.asarray(image)

    image = Image.fromarray(arr[:, :, 3])
    image = image.resize((32, 32))
    # transform split

    # if self.splitType == 'train':
    #   transform_train = transforms.Compose(
    #     [transforms.Resize((32, 32)),  # resizes the image so it can be perfect for our model.
    #      # transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
    #      transforms.RandomRotation(20),  # Rotates the image to a specified angle
    #      # transforms.RandomRotation(330),  # rotates in opp dir
    #      # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    #      # transforms.RandomVerticalFlip(),  # Performs actions like zooms, change shear angles.
    #      transforms.ToTensor(),  # convert the image to tensor so that it can work with torch
    #      transforms.Normalize((0.5,), (0.5,))
    #      ])
    #   image = transform_train(image)
    # elif self.splitType == 'test':
    #   transforms_test = transforms.Compose(
    #     [transforms.Resize((32, 32)),
    #      transforms.ToTensor(),
    #      transforms.Normalize((0.5,), (0.5,))
    #      ])
    #   image = transforms_test(image)

    # convert tensor image to numpy array of form (1, 1, 32, 32)
    image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]

    return image

  def __getitem__(self, item):

    imagePath = self.allImages[item]
    target = self.allTargets[item]
    imgClass = self.allClasses[target]
    image = self.process(config.basePath[self.splitType] + '/' + imgClass + '/image/' + imagePath)

    return image, target

  def __len__(self):
    return int(len(self.allImages))


def _worker_init_fn(worker_id):
  np.random.seed(worker_id)


def getDataLoader(type_='train'):

  return DataLoader(
    dataloader(splitType=type_),

    batch_size=config.batchSize[type_],
    num_workers=config.numWorkers[type_],
    worker_init_fn=_worker_init_fn,
    shuffle=True
  )
