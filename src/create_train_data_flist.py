import os
import argparse
import numpy as np
import pickle
from random import sample

# =============================================================================
# Define the index number of validation dataset
# =============================================================================

# 定義缺陷影像資料集的路徑
dataset_path = r"D:\joy\Structured_Light\SL_data\Normalize_Datast_Zero_DCE_with_First_Order"

path = os.path.join(dataset_path, 'First_Order')

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

# 讀取'First_Order'內部影像資料夾的影像存取成list
images = []
for root, dirs, files in os.walk(path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            images.append(os.path.join(root, file))


# 從20000張影像路徑所組成的list隨機抽取1000張影像作為驗證資料集
number = np.random.choice(20000, 1000, replace=False)
number = number.tolist()

txt_path = r'datasets\structured_light\val_num.txt'

with open(txt_path, 'wb') as fp:
    pickle.dump(number, fp)

# #%%
# with open(txt_path, 'rb') as fp:
#     list_1 = pickle.load(fp)

#%%
# =============================================================================
# Create three different type of data flist for training process
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--path',   type=str, help='path to the dataset')
parser.add_argument('--output_1', type=str, help='path to the file list')
parser.add_argument('--output_2', type=str, help='path to the file list')

args = parser.parse_args()

# 建立存放訓練影像資料路徑已集驗證影像資料路徑
os.makedirs(os.path.join(r'datasets\structured_light', 'train'))
os.makedirs(os.path.join(r'datasets\structured_light', 'val'))

# 分別按照3種不同的資料集進行讀取('First_Order', 'Ground_Truth', 'Normalize_Image')
# 在'train'以及'val'資料夾存放3種不同的資料集的影像路徑
for data_type in ['First_Order', 'Ground_Truth', 'Normalize_Image']:
    
    args.path = os.path.join(dataset_path, data_type)
    args.output_1 = os.path.join(r'datasets\structured_light', 'train', data_type+r'.flist')
    args.output_2 = os.path.join(r'datasets\structured_light', 'val', data_type+r'.flist')
    
    
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
    
    # 讀取未抽選為1000的影像路徑為訓練資料路徑
    images = []
    for root, dirs, files in os.walk(args.path):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1].upper() in ext:
                # 儲存路徑到train.flist檔
                images.append(os.path.join(root, file))

    images = sorted(images)
    train_images = images.copy()
    
    # 讀取抽選為1000的影像路徑為驗證資料路徑
    val_images   = []
    for val_num in number:
        val_image = images[val_num]    
        val_images.append(val_image)
        # 儲存路徑到val.flist檔
        train_images.remove(val_image)

    np.savetxt(args.output_1, train_images, fmt='%s')
    np.savetxt(args.output_2, val_images, fmt='%s')
    

    


