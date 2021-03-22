import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
random.seed(9001)
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
import argparse
import numpy as np
import torchvision
from utils import collate_self

class sketch_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join('/home/media/CVPR_2021/Sketch_Pooling/sketchPool/Sketch_Classification/TU_Berlin')
        # coordinate_path = r'E:\sketchPool\Sketch_Classification\TU_Berlin'
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        #get unique classes:
        get_all_classes, all_samples = [], []
        for x in list(self.Coordinate.keys()):
            get_all_classes.append(x.split('/')[0])
            all_samples.append(x)
        get_all_classes = list(set(get_all_classes))
        get_all_classes.sort()
        #################################################################
        #########  MAPPING DIctionary ###################################
        self.num2name, self.name2num = {}, {}
        for num, val in enumerate(get_all_classes):
            self.num2name[num] = val
            self.name2num[val] = num

        self.Train_Sketch, self.Test_Sketch = [], []
        for class_name in get_all_classes:
            per_class_data = np.array([x for x in all_samples if class_name == x.split('/')[0]])
            per_class_Train = per_class_data[random.sample(range(len(per_class_data)), int(len(per_class_data) * hp.splitTrain))]
            per_class_Test = set(per_class_data) - set(per_class_Train)
            self.Train_Sketch.extend(list(per_class_Train))
            self.Test_Sketch.extend(list(per_class_Test))

        print('Total Training Sample {}'.format(len(self.Train_Sketch)))
        print('Total Testing Sample {}'.format(len(self.Test_Sketch)))


        self.train_transform = get_ransform('Train')
        self.test_transform = get_ransform('Test')


    def __getitem__(self, item):

        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            vector_x = self.Coordinate[sketch_path]
            sketch_img, stroke_boxes = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
            sketch_img = self.train_transform(sketch_img)


            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'sketch_boxes': stroke_boxes, 'sketch_label': self.name2num[sketch_path.split('/')[0]]}


        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]

            sketch_img, stroke_boxes = rasterize_Sketch(vector_x)
            sketch_img = self.test_transform(Image.fromarray(sketch_img).convert('RGB'))

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'sketch_boxes': stroke_boxes, 'sketch_label': self.name2num[sketch_path.split('/')[0]]}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def get_dataloader(hp):

    dataset_Train  = sketch_Dataset(hp, mode = 'Train')
    dataset_Test = sketch_Dataset(hp, mode='Test')

    dataset_Test.Test_Sketch = dataset_Train.Test_Sketch
    dataset_Train.Test_Sketch = []
    dataset_Test.Train_Sketch = []

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=128, shuffle=False,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    return dataloader_Train, dataloader_Test


def get_ransform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(256)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(256)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)

# def get_ransform(type):
#     transform_list = []
#     if type is 'Train':
#         transform_list.extend([transforms.Resize(240), transforms.CenterCrop(224)])
#     elif type is 'Test':
#         transform_list.extend([transforms.Resize(224)])
#     transform_list.extend(
#         [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     return transforms.Compose(transform_list)


if __name__ == '__main__':

    ########## Module Testing ####################
    parser = argparse.ArgumentParser(description='Sketch_Classification')
    parser.add_argument('--dataset_name', type=str, default='TU-Berlin')
    hp = parser.parse_args()
    hp.batchsize = 64
    hp.nThreads = 4
    hp.splitTrain = 0.7
    dataset_Train = sketch_Dataset(hp, mode='Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                       num_workers=int(hp.nThreads))
    for data in dataloader_Train:
        print(data['sketch_img'].shape)
        torchvision.utils.save_image(data['sketch_img'], 'a.jpg', normalize=True)