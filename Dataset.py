import os
import lmdb # install lmdb by "pip install lmdb"
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
random.seed(9001)
import torchvision.transforms.functional as F
from rasterize import mydrawPNG_from_list, rasterize_Sketch
import argparse
import numpy as np
import torchvision
from utils import to_Five_Point



class Dataset_TUBerlin(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join('./TU_Berlin')

        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        total_del = 0
        all_keys = list(self.Coordinate.keys())
        for key in all_keys:
            if len(self.Coordinate[key]) > 300:
                del self.Coordinate[key]
                total_del += 1

        print('Total Number of samples deleted: {}'.format(total_del))
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
            sketch_img, sketch_points = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                sketch_points[:, 0] = -sketch_points[:, 0] + 256.
            sketch_img = self.train_transform(sketch_img)


            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_points': sketch_points,
                       'sketch_label': self.name2num[sketch_path.split('/')[0]]}


        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]

            sketch_img, sketch_points = rasterize_Sketch(vector_x)
            sketch_img = self.test_transform(Image.fromarray(sketch_img).convert('RGB'))

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_points': sketch_points,
                     'sketch_label': self.name2num[sketch_path.split('/')[0]]}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)


class Dataset_Quickdraw(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode

        if self.mode == 'Train':
            self.TrainData_ENV = lmdb.open('./QuickDraw/QuickDraw_TrainData',  max_readers=1,
                                           readonly=True, lock=False, readahead=False, meminit=False)
        elif self.mode == 'Test':
            self.TestData_ENV = lmdb.open('./QuickDraw/QuickDraw_TestData',  max_readers=1,
                                           readonly=True, lock=False, readahead=False, meminit=False)

        with open("./QuickDraw/QuickDraw_Keys.pickle", "rb") as handle:
            self.Train_keys, self.Valid_keys, self.Test_keys = pickle.load(handle)

        #get unique classes:
        get_all_classes, all_samples = [], []
        for x in self.Train_keys:
            get_all_classes.append(x.split('_')[0])
            all_samples.append(x)
        get_all_classes = list(set(get_all_classes))
        get_all_classes.sort()
        #################################################################
        #########  MAPPING DIctionary ###################################
        self.num2name, self.name2num = {}, {}
        for num, val in enumerate(get_all_classes):
            self.num2name[num] = val
            self.name2num[val] = num

        print('Total Training Sample {}'.format(len(self.Train_keys)))
        print('Total Testing Sample {}'.format(len(self.Test_keys)))


        self.train_transform = get_ransform('Train')
        self.test_transform = get_ransform('Test')

    def __getitem__(self, item):

        if self.mode == 'Train':
            with self.TrainData_ENV.begin(write=False) as txn:
                sketch_path = self.Train_keys[item]
                sample = txn.get(sketch_path.encode("ascii"))
                sketch_points = np.frombuffer(sample).reshape(-1, 3).copy()
                stroke_list = np.split(sketch_points[:, :2], np.where(sketch_points[:, 2])[0] + 1, axis=0)[:-1]
                sketch_img = mydrawPNG_from_list(stroke_list)

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                sketch_points[:, 0] = -sketch_points[:, 0] + 256.
            sketch_img = self.train_transform(sketch_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_points': sketch_points,
                       'sketch_label': self.name2num[sketch_path.split('_')[0]]}


        elif self.mode == 'Test':

            with self.TestData_ENV.begin(write=False) as txn:
                sketch_path = self.Test_keys[item]
                sample = txn.get(sketch_path.encode())
                sketch_points = np.frombuffer(sample).reshape(-1, 3).copy()
                stroke_list = np.split(sketch_points[:, :2], np.where(sketch_points[:, 2])[0] + 1, axis=0)[:-1]
                sketch_img = mydrawPNG_from_list(stroke_list)

                n_flip = random.random()
                if n_flip > 0.5:
                    sketch_img = F.hflip(sketch_img)
                    sketch_points[:, 0] = -sketch_points[:, 0] + 256.
                sketch_img = self.train_transform(sketch_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_points': sketch_points,
                      'sketch_label': self.name2num[sketch_path.split('_')[0]]}


        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_keys)
        elif self.mode == 'Test':
            return len(self.Test_keys)


def collate_self(batch):
    batch_mod = {'sketch_img': [], 'sketch_points': [],
                 'sketch_label': [], 'sketch_path': [], 'sketch_five_point': [], 'seq_len': [],
                 }

    max_len = max([len(x['sketch_points']) for x in batch])

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_label'].append(i_batch['sketch_label'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        five_point, len_seq = to_Five_Point(i_batch['sketch_points'], max_len)
        # First time step is [0, 0, 0, 0, 0] as start token.
        batch_mod['sketch_five_point'].append(torch.tensor(five_point))
        batch_mod['seq_len'].append(len_seq)

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])
    batch_mod['sketch_five_point'] = torch.stack(batch_mod['sketch_five_point'], dim=0)
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])

    return batch_mod


def get_dataloader(hp):

    if hp.dataset_name == 'TUBerlin':

        dataset_Train  = Dataset_TUBerlin(hp, mode = 'Train')
        dataset_Test = Dataset_TUBerlin(hp, mode='Test')

        dataset_Test.Test_Sketch = dataset_Train.Test_Sketch
        dataset_Train.Test_Sketch = []
        dataset_Test.Train_Sketch = []

    elif hp.dataset_name == 'QuickDraw':

        dataset_Train = Dataset_Quickdraw(hp, mode='Train')
        dataset_Test = Dataset_Quickdraw(hp, mode='Test')

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    return dataloader_Train, dataloader_Test



def get_ransform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(256)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(256)])
    # transform_list.extend(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transforms.Compose(transform_list)


if __name__ == '__main__':

    ########## Module Testing ####################
    parser = argparse.ArgumentParser(description='Sketch_Classification')
    parser.add_argument('--dataset_name', type=str, default='TUBerlin', help='TUBerlin vs QuickDraw')
    hp = parser.parse_args()
    hp.batchsize = 64
    hp.nThreads = 4
    hp.splitTrain = 0.7
    dataloader_Train, dataloader_Test = get_dataloader(hp)

    for data in dataloader_Train:
        print(data['sketch_img'].shape)
        torchvision.utils.save_image(data['sketch_img'], 'a.jpg', normalize=True)