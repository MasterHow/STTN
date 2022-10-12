import os
import cv2
import io
import glob
import scipy
import json
import zipfile
import random
import collections
import torch
import math
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from skimage.color import rgb2gray, gray2rgb
from core.utils import ZipReader, create_random_shape_with_random_motion, TrainZipReader, TestZipReader
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.num_local_frames = self.sample_length  # sample length是总输入的数量可能是局部也可能是非局部
        self.size = self.w, self.h = (args['w'], args['h'])

        if args['name'] != 'KITTI360-EX':
            # for youtube-vos and davis
            with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
                self.video_dict = json.load(f)
            self.video_names = list(self.video_dict.keys())
            self.dataset_name = args['name']
            if debug or split != 'train':
                self.video_names = self.video_names[:100]
        else:
            # 使用json读取训练list
            json_path = os.path.join(args['data_root'], 'train.json')
            with open(json_path, 'r') as f:
                self.video_dict = json.load(f)
            self.video_names = list(self.video_dict.keys())
            self.dataset_name = 'KITTI360-EX'

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        # try:
        if self.dataset_name != 'KITTI360-EX':
            item = self.load_item(index)
        elif self.dataset_name == 'KITTI360-EX':
            item = self.load_item_kitti(index)
        else:
            raise Exception('Unknown dataset.')
        # except:
        #     print('Loading error in video {}'.format(self.video_names[index]))
        #     item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors

    def load_item_kitti(self, index):
        video_name = self.video_names[index]
        # create masks

        # create sample index
        # FuseFormer只取5帧
        # 可能是随机的5帧也可能是连续的5帧
        selected_index = get_ref_index(self.video_dict[video_name], self.num_local_frames)

        # read video frames
        frames = []
        masks = []
        for idx in selected_index:

            video_path = os.path.join(self.args['data_root'],
                                      'JPEGImages',
                                      f'{video_name}.zip')

            img = TrainZipReader.imread(video_path, idx).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)

            # 对于KITTI360-EX数据集，读取zip中存储的mask
            mask_path = os.path.join(self.args['data_root'],
                                     'test_masks',
                                      f'{video_name}.zip')
            mask = TrainZipReader.imread(mask_path, idx)
            mask = mask.resize(self.size).convert('L')
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

        # normalizate, to tensors
        frames = GroupRandomHorizontalFlip()(frames)

        if self.dataset_name == 'KITTI360-EX':
            # 对于本地读取的mask 也需要随着frame翻转
            masks = GroupRandomHorizontalFlip()(masks)

        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
