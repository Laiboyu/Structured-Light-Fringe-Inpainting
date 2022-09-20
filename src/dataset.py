import os
import glob
import scipy
import torch
import random
import numpy as np
import cv2 as cv
import torchvision.transforms.functional as F
from torchvision import transforms, utils

from torch.utils.data import DataLoader
from PIL import Image
# from scipy.misc import imread
import imageio
from imageio import imread
from skimage import transform
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
# from .utils import create_mask

def normalize_func(self, a, b):
    "Return a-b if a>b, otherwise return a+b"
    if a > b:
        a = 1.0
        return a
    else:
        a = 0.0
        return a

class Random_Rescale_Rotate(object):
    '''
    以first-order、normalized、edge作為輸入影像，隨機旋轉以及縮放輸入影像進行增強。
    '''
    def __init__(self, image_size=256,
                 angle_range=[0, 360], 
                 scale_ratio=[0.6, 1.4]):
        
        assert isinstance(image_size,(int,tuple))
        ## 設立角度範圍(angle_range),在角度範圍內隨機選取一角度大小進行影像旋轉
        ## 設立規模因子(scale_ratio),在規模因子內隨機設定一規模大小進行影像縮放
        self.angle_range = angle_range
        self.scale_ratio = scale_ratio
        self.image_size = image_size
        
        
    def __call__(self, sample):
        
        image, label, fo_image = sample[0], sample[1], sample[2]
        
        shape = image.shape
        
        ## randomly_select_parameter
        random_angle = random.randrange(self.angle_range[0], self.angle_range[1])
        random_scale = random.uniform(self.scale_ratio[0], self.scale_ratio[1])
        
        ## randomly_rotate_image
        image = transform.rotate(image, random_angle)
        label = transform.rotate(label, random_angle)
        fo_image = transform.rotate(fo_image, random_angle)

        ## randomly_resize_image        
        if random_scale > 1:
            # print("random_number %f is lager than 1" %random_scale)
            
            ## 設定新影像的尺寸大小
            new_shape = [round(shape[0]*random_scale), round(shape[1]*random_scale)]
            start_point = [round((new_shape[0]-shape[0])/2), round((new_shape[1]-shape[1])/2)]
            
            # 設定初始座標點來置放縮放後的影像
            resized_image = cv.resize(image, new_shape, interpolation=cv.INTER_LINEAR)
            if len(resized_image.shape) != 3:
                resized_image = np.expand_dims(resized_image, axis=2)
            new_image = resized_image[start_point[0]:start_point[0]+self.image_size, start_point[1]:start_point[1]+self.image_size, :]
            
            resized_label = cv.resize(label, new_shape, interpolation=cv.INTER_LINEAR)
            new_label = resized_label[start_point[0]:start_point[0]+self.image_size, start_point[1]:start_point[1]+self.image_size]
            
            resized_fo_image = cv.resize(fo_image, new_shape, interpolation=cv.INTER_LINEAR)
            if len(resized_fo_image.shape) != 3:
                resized_fo_image = np.expand_dims(resized_fo_image, axis=2)
            new_fo_image = resized_fo_image[start_point[0]:start_point[0]+self.image_size, start_point[1]:start_point[1]+self.image_size, :]
            
                    
        else:
            # print("random_number %f is smaller than 1" %random_scale)
            new_image = np.zeros(shape, dtype=np.float32)
            new_label = np.zeros((shape[0], shape[1]), dtype=np.float32)
            new_fo_image = np.zeros(shape, dtype=np.float32)

            new_shape = [round(shape[0]*random_scale), round(shape[1]*random_scale)]
            start_point = [round((shape[0]-new_shape[0])/2), round((shape[1]-new_shape[1])/2)]
            
            resized_image = cv.resize(image, new_shape, interpolation=cv.INTER_LINEAR)
            if len(resized_image.shape) != 3:
                resized_image = np.expand_dims(resized_image, axis=2)
            new_image[start_point[0]:start_point[0]+new_shape[0], start_point[1]:start_point[1]+new_shape[1], :] = resized_image

            resized_label = cv.resize(label, new_shape, interpolation=cv.INTER_LINEAR)
            new_label[start_point[0]:start_point[0]+new_shape[0], start_point[1]:start_point[1]+new_shape[1]] = resized_label
            
            resized_fo_image = cv.resize(fo_image, new_shape, interpolation=cv.INTER_LINEAR)
            if len(resized_fo_image.shape) != 3:
                resized_fo_image = np.expand_dims(resized_fo_image, axis=2)
            new_fo_image[start_point[0]:start_point[0]+new_shape[0], start_point[1]:start_point[1]+new_shape[1], :] = resized_fo_image
            
        # Normalize_func = np.vectorize(normalize_func, otypes=[np.float32])
        # new_label = Normalize_func(new_label, 0.5)
        
        # del image, label, random_angle, random_scale, resized_image, resized_label
        # del new_shape, start_point, Normalize_func
        
        sample = [new_image, new_label, new_fo_image]

        return sample


class Dataset(torch.utils.data.Dataset):
    """
    讀取由影像路徑組成的flist檔案，建立由tensor所組成的影像資料庫，
    一組資料包含著 input:(img、fo_imge、edge_img)，label:(gt_img、edge_gt)
    """
    def __init__(self, config, flist, gt_flist, fo_flist, transform=None, **kwargs):
        super(Dataset, self).__init__(**kwargs)
        self.transform = transform
        self.data = self.load_flist(flist)
        self.gt_data = self.load_flist(gt_flist)
        self.fo_data = self.load_flist(fo_flist)
         
        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size
        
        # load image
        img = imread(self.data[index])
        # load ground_truth image
        gt_img = imread(self.gt_data[index])
        # load first_order image
        fo_img = imread(self.fo_data[index])
        
        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)     
            fo_img = gray2rgb(fo_img)  

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)
            gt_img = self.resize(gt_img, size, size)
            fo_img = self.resize(fo_img, size, size)
            
        # data transform(影像增強)
        if self.transform:
            sample = [img, gt_img, fo_img]
            sample = self.transform(sample) 
            img = sample[0].astype(np.float64)
            gt_img  = sample[1].astype(np.float64)
            fo_img  = sample[2].astype(np.float64)
        else:
            gt_img = (gt_img/255.0).astype(np.float64)
        
        # create grayscale image
        gray_img = rgb2gray(img)
        gray_fo_img = rgb2gray(fo_img)

        # load edge
        edge_img = self.load_edge(gray_img, index)
        edge_gt  = self.load_edge(gt_img, index)
        
        return  self.to_tensor(gray_fo_img), self.to_tensor(gray_img), self.to_tensor(edge_img), self.to_tensor(gt_img), self.to_tensor(edge_gt)

    def load_edge(self, img, index):
        sigma = self.sigma

        # canny
        if sigma == -1:
            return np.zeros(img.shape).astype(np.float)

        # random sigma
        if sigma == 0:
            sigma = random.randint(1, 4)

        canny_img = canny(img, sigma=sigma)

        return canny_img


    def to_tensor(self, img):
        # 將plt.image轉換為tensor檔案
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]
        # print(img.shape)

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # img = imageio.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize([height, width]))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        #　建立迭代資料由資料庫中隨機選取資料進行運算
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class Prediction_Dataset(torch.utils.data.Dataset):
    """
    讀取由影像路徑組成的flist檔案，建立由tensor所組成的影像資料庫，
    一組資料包含著 input:(img、fo_imge、edge_img)
    """
    def __init__(self, config, flist, fo_flist, transform=None, **kwargs):
        super(Prediction_Dataset, self).__init__(**kwargs)
        self.transform = transform
        self.data = self.load_flist(flist)
        self.fo_data = self.load_flist(fo_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size
        
        # load image
        img = imread(self.data[index])
        # load first_order image
        fo_img = imread(self.fo_data[index])
        
        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)     
            fo_img = gray2rgb(fo_img)  

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)
            fo_img = self.resize(fo_img, size, size)
            
        # data transform 
        if self.transform:
            sample = [img, fo_img]
            sample = self.transform(sample) 
            img = sample[0].astype(np.float64)
            fo_img  = sample[1].astype(np.float64)
        
        # create grayscale image
        gray_img = rgb2gray(img)
        gray_fo_img = rgb2gray(fo_img)

        # load edge
        edge_img = self.load_edge(gray_img, index)

        return  self.to_tensor(gray_fo_img), self.to_tensor(gray_img), self.to_tensor(edge_img)
    
    
    def load_edge(self, img, index):
        sigma = self.sigma

        if sigma == -1:
            return np.zeros(img.shape).astype(np.float)
        # random sigma
        if sigma == 0:
            sigma = random.randint(1, 4)

        canny_img = canny(img, sigma=sigma)

        return canny_img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]
        # print(img.shape)

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # img = imageio.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize([height, width]))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item